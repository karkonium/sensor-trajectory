def map_sensor_to_original(sensor_coords, combined_shape, horizontal_concat=True):
    """
    Map sensor coordinates from a combined (augmented) domain back to the original domain.
    
    Parameters:
      sensor_coords : Each row is a coordinate [i, j] in the combined domain.
      combined_shape : The shape (n_rows, n_cols) of the combined data.
          For horizontal concatenation, this should be (nx, 2*ny_orig).
          For vertical concatenation, this should be (2*nx_orig, ny).
      mode : 'horizontal' or 'vertical'. Default is 'horizontal'.
    
    Returns:
      mapped_coords : The sensor coordinates mapped back to the original grid.
          
    Explanation:
      - In horizontal mode, if a sensor’s column index j is ≥ ny_orig,
        then its original column index is j - ny_orig (while the row index stays the same).
      - In vertical mode, if a sensor’s row index i is ≥ nx_orig,
        then its original row index is i - nx_orig (while the column index stays the same).
    """
    mapped_coords = sensor_coords.copy() 
    
    if horizontal_concat:
        # combined_shape[1] is 2 * ny_orig.
        ny_combined = combined_shape[1]
        ny_orig = ny_combined // 2
        for idx, (i, j) in enumerate(sensor_coords):
            if j >= ny_orig:
                mapped_coords[idx, 1] = j - ny_orig
            # else: keep the same
    else:
        # combined_shape[0] is 2 * nx_orig.
        nx_combined = combined_shape[0]
        nx_orig = nx_combined // 2
        for idx, (i, j) in enumerate(sensor_coords):
            if i >= nx_orig:
                mapped_coords[idx, 0] = i - nx_orig
    
    return mapped_coords