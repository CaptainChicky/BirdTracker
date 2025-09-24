// benches/ray_casting.rs
// Performance benchmarks for the voxel tracking system

use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId};
use nalgebra::Vector3;
use voxel_tracker::{VoxelGrid, cast_ray_dda, MotionMask, Frame, CameraParams, create_rotation_matrix, process_frame_motion};

fn benchmark_ray_casting(c: &mut Criterion) {
    let mut group = c.benchmark_group("ray_casting");
    
    for grid_size in [50, 100, 200].iter() {
        let grid = VoxelGrid::new(*grid_size, 0.1, Vector3::new(0.0, 0.0, 0.0));
        let origin = Vector3::new(-5.0, 0.0, 0.0);
        let direction = Vector3::new(1.0, 0.0, 0.0);
        
        group.bench_with_input(
            BenchmarkId::new("cast_ray_dda", grid_size),
            grid_size,
            |b, _| {
                b.iter(|| {
                    cast_ray_dda(
                        black_box(&origin),
                        black_box(&direction),
                        black_box(&grid),
                        black_box(20.0),
                    )
                })
            },
        );
    }
    group.finish();
}

fn benchmark_motion_detection(c: &mut Criterion) {
    let mut group = c.benchmark_group("motion_detection");
    
    for size in [480, 720, 1080].iter() {
        let width = (size * 16) / 9; // 16:9 aspect ratio
        let height = *size;
        
        let frame1_data = vec![0.5; (width * height) as usize];
        let frame2_data = vec![0.6; (width * height) as usize];
        
        let frame1 = Frame { 
            data: frame1_data, 
            width: width as u32, 
            height: height as u32 
        };
        let frame2 = Frame { 
            data: frame2_data, 
            width: width as u32, 
            height: height as u32 
        };
        
        group.bench_with_input(
            BenchmarkId::new("motion_detection", size),
            size,
            |b, _| {
                b.iter(|| {
                    MotionMask::from_frames(
                        black_box(&frame1),
                        black_box(&frame2),
                        black_box(0.1),
                    )
                })
            },
        );
    }
    group.finish();
}

fn benchmark_frame_processing(c: &mut Criterion) {
    let mut group = c.benchmark_group("frame_processing");
    
    let width = 640;
    let height = 480;
    let grid_size = 100;
    
    // Create test data
    let mut motion_data = vec![false; (width * height) as usize];
    // Set 10% of pixels to have motion
    for i in (0..(width * height) as usize).step_by(10) {
        motion_data[i] = true;
    }
    
    let motion_mask = MotionMask {
        data: motion_data,
        width: width as u32,
        height: height as u32,
    };
    
    let frame_data = vec![0.7; (width * height) as usize];
    let frame = Frame {
        data: frame_data,
        width: width as u32,
        height: height as u32,
    };
    
    let camera = CameraParams {
        position: Vector3::new(0.0, 0.0, 0.0),
        rotation: create_rotation_matrix(0.0, 0.0, 0.0),
        fov_radians: 60.0_f32.to_radians(),
        width: width as u32,
        height: height as u32,
    };
    
    let mut voxel_grid = VoxelGrid::new(grid_size, 0.1, Vector3::new(0.0, 0.0, 5.0));
    
    group.bench_function("process_frame_motion", |b| {
        b.iter(|| {
            process_frame_motion(
                black_box(&motion_mask),
                black_box(&frame),
                black_box(&camera),
                black_box(&mut voxel_grid),
                black_box(10.0),
            );
        })
    });
    
    group.finish();
}

fn benchmark_voxel_operations(c: &mut Criterion) {
    let mut group = c.benchmark_group("voxel_operations");
    
    for grid_size in [100, 200, 400].iter() {
        let grid = VoxelGrid::new(*grid_size, 0.1, Vector3::new(0.0, 0.0, 0.0));
        
        group.bench_with_input(
            BenchmarkId::new("world_to_voxel", grid_size),
            grid_size,
            |b, _| {
                b.iter(|| {
                    grid.world_to_voxel(black_box(&Vector3::new(1.0, 2.0, 3.0)))
                })
            },
        );
    }
    
    group.finish();
}

criterion_group!(
    benches,
    benchmark_ray_casting,
    benchmark_motion_detection,
    benchmark_frame_processing,
    benchmark_voxel_operations
);
criterion_main!(benches);