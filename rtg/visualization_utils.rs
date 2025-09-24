// visualization.rs
// Utilities for visualizing and exporting voxel data

use crate::VoxelGrid;
use nalgebra::Vector3;
use std::fs::File;
use std::io::{Write, BufWriter};
use std::path::Path;

#[derive(Debug, Clone)]
pub struct VoxelPoint {
    pub position: Vector3<f32>,
    pub intensity: f32,
    pub voxel_coords: (usize, usize, usize),
}

pub struct VoxelExporter;

impl VoxelExporter {
    /// Export voxel data as PLY point cloud for visualization in MeshLab, CloudCompare, etc.
    pub fn export_to_ply<P: AsRef<Path>>(
        voxel_grid: &VoxelGrid,
        path: P,
        threshold_percentile: f32,
    ) -> Result<(), std::io::Error> {
        let points = Self::extract_significant_points(voxel_grid, threshold_percentile);
        
        let file = File::create(path)?;
        let mut writer = BufWriter::new(file);
        
        // Write PLY header
        writeln!(writer, "ply")?;
        writeln!(writer, "format ascii 1.0")?;
        writeln!(writer, "element vertex {}", points.len())?;
        writeln!(writer, "property float x")?;
        writeln!(writer, "property float y")?;
        writeln!(writer, "property float z")?;
        writeln!(writer, "property float intensity")?;
        writeln!(writer, "property uchar red")?;
        writeln!(writer, "property uchar green")?;
        writeln!(writer, "property uchar blue")?;
        writeln!(writer, "end_header")?;
        
        // Write vertices with color mapping based on intensity
        for point in points {
            let (r, g, b) = Self::intensity_to_color(point.intensity, 0.0, 1.0);
            writeln!(
                writer,
                "{:.6} {:.6} {:.6} {:.6} {} {} {}",
                point.position.x, point.position.y, point.position.z,
                point.intensity,
                r, g, b
            )?;
        }
        
        writer.flush()?;
        Ok(())
    }
    
    /// Export voxel data as CSV for analysis in spreadsheet applications
    pub fn export_to_csv<P: AsRef<Path>>(
        voxel_grid: &VoxelGrid,
        path: P,
        threshold_percentile: f32,
    ) -> Result<(), std::io::Error> {
        let points = Self::extract_significant_points(voxel_grid, threshold_percentile);
        
        let file = File::create(path)?;
        let mut writer = BufWriter::new(file);
        
        writeln!(writer, "x,y,z,intensity,voxel_x,voxel_y,voxel_z")?;
        
        for point in points {
            writeln!(
                writer,
                "{:.6},{:.6},{:.6},{:.6},{},{},{}",
                point.position.x, point.position.y, point.position.z,
                point.intensity,
                point.voxel_coords.0, point.voxel_coords.1, point.voxel_coords.2
            )?;
        }
        
        writer.flush()?;
        Ok(())
    }
    
    /// Export 2D slice of voxel grid as grayscale image
    pub fn export_slice_as_image<P: AsRef<Path>>(
        voxel_grid: &VoxelGrid,
        path: P,
        slice_axis: SliceAxis,
        slice_index: usize,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let (width, height, slice_data) = Self::extract_slice(voxel_grid, slice_axis, slice_index);
        
        // Normalize slice data to 0-255 range
        let max_val = slice_data.iter().fold(0.0f32, |a, &b| a.max(b));
        let min_val = slice_data.iter().fold(f32::INFINITY, |a, &b| a.min(b));
        let range = max_val - min_val;
        
        let normalized: Vec<u8> = if range > 0.0 {
            slice_data.iter()
                .map(|&val| ((val - min_val) / range * 255.0) as u8)
                .collect()
        } else {
            vec![0u8; slice_data.len()]
        };
        
        let img_buffer = image::ImageBuffer::from_raw(width, height, normalized)
            .ok_or("Failed to create image buffer")?;
        
        img_buffer.save(path)?;
        Ok(())
    }
    
    fn extract_significant_points(voxel_grid: &VoxelGrid, threshold_percentile: f32) -> Vec<VoxelPoint> {
        let mut sorted_values: Vec<f32> = voxel_grid.data.iter()
            .filter(|&&x| x > 0.0)
            .cloned()
            .collect();
        
        if sorted_values.is_empty() {
            return Vec::new();
        }
        
        sorted_values.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let threshold_idx = ((sorted_values.len() as f32) * (threshold_percentile / 100.0)) as usize;
        let threshold = sorted_values.get(threshold_idx.min(sorted_values.len() - 1))
            .copied()
            .unwrap_or(0.0);
        
        let mut points = Vec::new();
        let half_extent = (voxel_grid.size as f32 * voxel_grid.voxel_size) / 2.0;
        let grid_min = voxel_grid.center - Vector3::new(half_extent, half_extent, half_extent);
        
        for x in 0..voxel_grid.size {
            for y in 0..voxel_grid.size {
                for z in 0..voxel_grid.size {
                    let idx = voxel_grid.get_index(x, y, z);
                    let intensity = voxel_grid.data[idx];
                    
                    if intensity > threshold {
                        let world_pos = Vector3::new(
                            grid_min.x + (x as f32 + 0.5) * voxel_grid.voxel_size,
                            grid_min.y + (y as f32 + 0.5) * voxel_grid.voxel_size,
                            grid_min.z + (z as f32 + 0.5) * voxel_grid.voxel_size,
                        );
                        
                        points.push(VoxelPoint {
                            position: world_pos,
                            intensity,
                            voxel_coords: (x, y, z),
                        });
                    }
                }
            }
        }
        
        points
    }
    
    fn intensity_to_color(intensity: f32, min_val: f32, max_val: f32) -> (u8, u8, u8) {
        let normalized = if max_val > min_val {
            ((intensity - min_val) / (max_val - min_val)).clamp(0.0, 1.0)
        } else {
            0.0
        };
        
        // Hot colormap: black -> red -> yellow -> white
        if normalized < 0.33 {
            let t = normalized / 0.33;
            (
                (255.0 * t) as u8,
                0,
                0,
            )
        } else if normalized < 0.66 {
            let t = (normalized - 0.33) / 0.33;
            (
                255,
                (255.0 * t) as u8,
                0,
            )
        } else {
            let t = (normalized - 0.66) / 0.34;
            (
                255,
                255,
                (255.0 * t) as u8,
            )
        }
    }
    
    fn extract_slice(voxel_grid: &VoxelGrid, axis: SliceAxis, index: usize) -> (u32, u32, Vec<f32>) {
        match axis {
            SliceAxis::X => {
                let mut data = Vec::with_capacity(voxel_grid.size * voxel_grid.size);
                for z in 0..voxel_grid.size {
                    for y in 0..voxel_grid.size {
                        let idx = voxel_grid.get_index(index, y, z);
                        data.push(voxel_grid.data[idx]);
                    }
                }
                (voxel_grid.size as u32, voxel_grid.size as u32, data)
            },
            SliceAxis::Y => {
                let mut data = Vec::with_capacity(voxel_grid.size * voxel_grid.size);
                for z in 0..voxel_grid.size {
                    for x in 0..voxel_grid.size {
                        let idx = voxel_grid.get_index(x, index, z);
                        data.push(voxel_grid.data[idx]);
                    }
                }
                (voxel_grid.size as u32, voxel_grid.size as u32, data)
            },
            SliceAxis::Z => {
                let mut data = Vec::with_capacity(voxel_grid.size * voxel_grid.size);
                for y in 0..voxel_grid.size {
                    for x in 0..voxel_grid.size {
                        let idx = voxel_grid.get_index(x, y, index);
                        data.push(voxel_grid.data[idx]);
                    }
                }
                (voxel_grid.size as u32, voxel_grid.size as u32, data)
            },
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub enum SliceAxis {
    X,
    Y,
    Z,
}

/// Statistics and analysis utilities
pub struct VoxelAnalyzer;

impl VoxelAnalyzer {
    pub fn compute_statistics(voxel_grid: &VoxelGrid) -> VoxelStatistics {
        let non_zero_values: Vec<f32> = voxel_grid.data.iter()
            .filter(|&&x| x > 0.0)
            .cloned()
            .collect();
        
        if non_zero_values.is_empty() {
            return VoxelStatistics::default();
        }
        
        let total_count = voxel_grid.data.len();
        let non_zero_count = non_zero_values.len();
        let sum: f32 = non_zero_values.iter().sum();
        let mean = sum / non_zero_count as f32;
        
        let min_value = non_zero_values.iter().fold(f32::INFINITY, |a, &b| a.min(b));
        let max_value = non_zero_values.iter().fold(0.0f32, |a, &b| a.max(b));
        
        // Compute variance and standard deviation
        let variance = non_zero_values.iter()
            .map(|&x| (x - mean).powi(2))
            .sum::<f32>() / non_zero_count as f32;
        let std_dev = variance.sqrt();
        
        // Compute percentiles
        let mut sorted = non_zero_values.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
        
        let p50_idx = sorted.len() / 2;
        let p90_idx = (sorted.len() as f32 * 0.9) as usize;
        let p95_idx = (sorted.len() as f32 * 0.95) as usize;
        let p99_idx = (sorted.len() as f32 * 0.99) as usize;
        
        VoxelStatistics {
            total_voxels: total_count,
            non_zero_voxels: non_zero_count,
            occupancy_percentage: (non_zero_count as f32 / total_count as f32) * 100.0,
            min_value,
            max_value,
            mean,
            std_dev,
            total_intensity: sum,
            percentile_50: sorted.get(p50_idx).copied().unwrap_or(0.0),
            percentile_90: sorted.get(p90_idx).copied().unwrap_or(0.0),
            percentile_95: sorted.get(p95_idx).copied().unwrap_or(0.0),
            percentile_99: sorted.get(p99_idx).copied().unwrap_or(0.0),
        }
    }
    
    pub fn find_connected_components(
        voxel_grid: &VoxelGrid, 
        threshold: f32,
        min_component_size: usize,
    ) -> Vec<ConnectedComponent> {
        let mut visited = vec![false; voxel_grid.data.len()];
        let mut components = Vec::new();
        
        for x in 0..voxel_grid.size {
            for y in 0..voxel_grid.size {
                for z in 0..voxel_grid.size {
                    let idx = voxel_grid.get_index(x, y, z);
                    
                    if !visited[idx] && voxel_grid.data[idx] > threshold {
                        let component = Self::flood_fill(
                            voxel_grid, 
                            &mut visited, 
                            (x, y, z), 
                            threshold
                        );
                        
                        if component.voxels.len() >= min_component_size {
                            components.push(component);
                        }
                    }
                }
            }
        }
        
        // Sort by total intensity (largest first)
        components.sort_by(|a, b| b.total_intensity.partial_cmp(&a.total_intensity).unwrap());
        
        components
    }
    
    fn flood_fill(
        voxel_grid: &VoxelGrid,
        visited: &mut [bool],
        start: (usize, usize, usize),
        threshold: f32,
    ) -> ConnectedComponent {
        let mut stack = vec![start];
        let mut component_voxels = Vec::new();
        let mut total_intensity = 0.0;
        
        while let Some((x, y, z)) = stack.pop() {
            let idx = voxel_grid.get_index(x, y, z);
            
            if visited[idx] || voxel_grid.data[idx] <= threshold {
                continue;
            }
            
            visited[idx] = true;
            component_voxels.push((x, y, z));
            total_intensity += voxel_grid.data[idx];
            
            // Add neighbors to stack
            for (dx, dy, dz) in [
                (-1, 0, 0), (1, 0, 0), (0, -1, 0), 
                (0, 1, 0), (0, 0, -1), (0, 0, 1)
            ].iter() {
                let nx = x as i32 + dx;
                let ny = y as i32 + dy;
                let nz = z as i32 + dz;
                
                if nx >= 0 && nx < voxel_grid.size as i32 &&
                   ny >= 0 && ny < voxel_grid.size as i32 &&
                   nz >= 0 && nz < voxel_grid.size as i32 {
                    stack.push((nx as usize, ny as usize, nz as usize));
                }
            }
        }
        
        // Compute centroid
        let centroid = Self::compute_centroid(voxel_grid, &component_voxels);
        
        ConnectedComponent {
            voxels: component_voxels,
            total_intensity,
            centroid,
        }
    }
    
    fn compute_centroid(voxel_grid: &VoxelGrid, voxels: &[(usize, usize, usize)]) -> Vector3<f32> {
        let half_extent = (voxel_grid.size as f32 * voxel_grid.voxel_size) / 2.0;
        let grid_min = voxel_grid.center - Vector3::new(half_extent, half_extent, half_extent);
        
        let mut weighted_sum = Vector3::new(0.0, 0.0, 0.0);
        let mut total_weight = 0.0;
        
        for &(x, y, z) in voxels {
            let idx = voxel_grid.get_index(x, y, z);
            let weight = voxel_grid.data[idx];
            
            let world_pos = Vector3::new(
                grid_min.x + (x as f32 + 0.5) * voxel_grid.voxel_size,
                grid_min.y + (y as f32 + 0.5) * voxel_grid.voxel_size,
                grid_min.z + (z as f32 + 0.5) * voxel_grid.voxel_size,
            );
            
            weighted_sum += world_pos * weight;
            total_weight += weight;
        }
        
        if total_weight > 0.0 {
            weighted_sum / total_weight
        } else {
            Vector3::new(0.0, 0.0, 0.0)
        }
    }
}

#[derive(Debug, Default)]
pub struct VoxelStatistics {
    pub total_voxels: usize,
    pub non_zero_voxels: usize,
    pub occupancy_percentage: f32,
    pub min_value: f32,
    pub max_value: f32,
    pub mean: f32,
    pub std_dev: f32,
    pub total_intensity: f32,
    pub percentile_50: f32,
    pub percentile_90: f32,
    pub percentile_95: f32,
    pub percentile_99: f32,
}

#[derive(Debug)]
pub struct ConnectedComponent {
    pub voxels: Vec<(usize, usize, usize)>,
    pub total_intensity: f32,
    pub centroid: Vector3<f32>,
}

impl std::fmt::Display for VoxelStatistics {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f,
            "Voxel Statistics:\n\
             Total voxels: {}\n\
             Non-zero voxels: {} ({:.2}%)\n\
             Value range: {:.6} to {:.6}\n\
             Mean: {:.6}, Std Dev: {:.6}\n\
             Total intensity: {:.2}\n\
             Percentiles - 50th: {:.6}, 90th: {:.6}, 95th: {:.6}, 99th: {:.6}",
            self.total_voxels,
            self.non_zero_voxels,
            self.occupancy_percentage,
            self.min_value,
            self.max_value,
            self.mean,
            self.std_dev,
            self.total_intensity,
            self.percentile_50,
            self.percentile_90,
            self.percentile_95,
            self.percentile_99
        )
    }
}