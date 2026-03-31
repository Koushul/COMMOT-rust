use std::num::NonZeroU64;

use bytemuck::{Pod, Zeroable};
use thiserror::Error;
use wgpu::util::DeviceExt;

const SHADER: &str = include_str!("shaders/pairwise_distance.wgsl");

#[derive(Debug, Error)]
pub enum GpuDistanceError {
    #[error("no suitable GPU adapter: {0}")]
    NoAdapter(String),
    #[error("request_device: {0}")]
    RequestDevice(String),
    #[error("buffer size {need} exceeds device limit {limit}")]
    BufferTooLarge { need: u64, limit: u64 },
    #[error("buffer map: {0}")]
    MapFailed(String),
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Params {
    n: u32,
    dim: u32,
    output_squared: u32,
    _pad: u32,
}

pub async fn spatial_pairwise_costs_f64_async(
    spatial: &[f64],
    n: usize,
    dim: usize,
    output_squared_euclidean: bool,
) -> Result<Vec<f64>, GpuDistanceError> {
    if n == 0 {
        return Ok(Vec::new());
    }
    if spatial.len() != n * dim {
        return Err(GpuDistanceError::RequestDevice(format!(
            "spatial len {} != n*dim {}",
            spatial.len(),
            n * dim
        )));
    }

    let nn = n as u64;
    let out_elems = nn
        .checked_mul(nn)
        .ok_or_else(|| GpuDistanceError::RequestDevice("n*n overflow".into()))?;
    let out_bytes = out_elems
        .checked_mul(4)
        .ok_or_else(|| GpuDistanceError::RequestDevice("output bytes overflow".into()))?;
    let spatial_elems = (n * dim) as u64;
    let spatial_bytes = spatial_elems
        .checked_mul(4)
        .ok_or_else(|| GpuDistanceError::RequestDevice("spatial bytes overflow".into()))?;

    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
        backends: wgpu::Backends::all(),
        ..wgpu::InstanceDescriptor::new_without_display_handle()
    });

    let adapter = instance
        .request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: None,
            force_fallback_adapter: false,
        })
        .await
        .map_err(|e| GpuDistanceError::NoAdapter(e.to_string()))?;

    let mut limits = adapter.limits().clone();
    let max_bind = out_bytes.max(spatial_bytes);
    limits.max_storage_buffer_binding_size = limits.max_storage_buffer_binding_size.max(max_bind);
    let need_total = out_bytes + spatial_bytes + 256;
    limits.max_buffer_size = limits.max_buffer_size.max(need_total);

    let (device, queue) = adapter
        .request_device(&wgpu::DeviceDescriptor {
            label: Some("commot-distance"),
            required_features: wgpu::Features::empty(),
            required_limits: limits,
            experimental_features: wgpu::ExperimentalFeatures::default(),
            memory_hints: wgpu::MemoryHints::default(),
            trace: wgpu::Trace::default(),
        })
        .await
        .map_err(|e| GpuDistanceError::RequestDevice(e.to_string()))?;

    if need_total > device.limits().max_buffer_size {
        return Err(GpuDistanceError::BufferTooLarge {
            need: need_total,
            limit: device.limits().max_buffer_size,
        });
    }
    if out_bytes > device.limits().max_storage_buffer_binding_size
        || spatial_bytes > device.limits().max_storage_buffer_binding_size
    {
        return Err(GpuDistanceError::BufferTooLarge {
            need: max_bind,
            limit: device.limits().max_storage_buffer_binding_size,
        });
    }

    let spatial_f32: Vec<f32> = spatial.iter().map(|&x| x as f32).collect();
    let spatial_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("spatial"),
        contents: bytemuck::cast_slice(&spatial_f32),
        usage: wgpu::BufferUsages::STORAGE,
    });

    let out_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("dist_out"),
        size: out_bytes,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });

    let params = Params {
        n: n as u32,
        dim: dim as u32,
        output_squared: u32::from(output_squared_euclidean),
        _pad: 0,
    };
    let uniform_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("params"),
        contents: bytemuck::bytes_of(&params),
        usage: wgpu::BufferUsages::UNIFORM,
    });

    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("pairwise_distance"),
        source: wgpu::ShaderSource::Wgsl(SHADER.into()),
    });

    let bind_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("dist_bgl"),
        entries: &[
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: NonZeroU64::new(spatial_bytes.max(4)),
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 1,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: NonZeroU64::new(out_bytes.max(4)),
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 2,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: NonZeroU64::new(std::mem::size_of::<Params>() as u64),
                },
                count: None,
            },
        ],
    });

    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("dist_pl"),
        bind_group_layouts: &[Some(&bind_layout)],
        immediate_size: 0,
    });

    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("dist_pipeline"),
        layout: Some(&pipeline_layout),
        module: &shader,
        entry_point: Some("main"),
        compilation_options: Default::default(),
        cache: None,
    });

    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("dist_bg"),
        layout: &bind_layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: spatial_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: out_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: uniform_buf.as_entire_binding(),
            },
        ],
    });

    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("dist_enc"),
    });
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("dist_pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        let gx = (n as u32 + 15) / 16;
        let gy = (n as u32 + 15) / 16;
        pass.dispatch_workgroups(gx, gy, 1);
    }

    let readback = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("readback"),
        size: out_bytes,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    encoder.copy_buffer_to_buffer(&out_buf, 0, &readback, 0, out_bytes);
    queue.submit(Some(encoder.finish()));

    let slice = readback.slice(..);
    let (tx, rx) = std::sync::mpsc::channel::<std::result::Result<(), wgpu::BufferAsyncError>>();
    slice.map_async(wgpu::MapMode::Read, move |r| {
        let _ = tx.send(r);
    });
    device
        .poll(wgpu::PollType::wait_indefinitely())
        .map_err(|e| GpuDistanceError::MapFailed(format!("{e:?}")))?;
    rx.recv()
        .map_err(|_| GpuDistanceError::MapFailed("channel".into()))?
        .map_err(|e| GpuDistanceError::MapFailed(format!("{e:?}")))?;

    let data = slice.get_mapped_range();
    let out_f32: &[f32] = bytemuck::cast_slice(&data);
    let out: Vec<f64> = out_f32.iter().map(|&x| x as f64).collect();
    drop(data);
    readback.unmap();

    Ok(out)
}

pub fn spatial_pairwise_costs_f64(
    spatial: &[f64],
    n: usize,
    dim: usize,
    output_squared_euclidean: bool,
) -> Result<Vec<f64>, GpuDistanceError> {
    pollster::block_on(spatial_pairwise_costs_f64_async(
        spatial,
        n,
        dim,
        output_squared_euclidean,
    ))
}
