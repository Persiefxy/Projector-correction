import numpy as np

# Gray code decoding
def computeNumberOfPatternImages(width, height):
    assert width > 0 and height > 0
    n_cols = int(np.ceil(np.log2(width))) * 2
    n_rows = int(np.ceil(np.log2(height))) * 2
    return n_cols, n_rows

def gray_decoder(gray):
    dec = 0
    tmp = gray[0]
    if tmp:
        dec += 2 ** (len(gray) - 1)
    for i in range(1, len(gray)):
        tmp = tmp ^ gray[i]
        if tmp:
            dec += 2 ** (len(gray) - i - 1)
    return dec

# Filter out matching points and record
def gray_decode(images, parameters):
    pro_size = (1920, 1080) # Projector image plane size
    n_cols, n_rows = computeNumberOfPatternImages(pro_size[0], pro_size[1])
    n_patterns = n_cols + n_rows
    black_img, white_img = images[n_patterns:n_patterns + 2]
    shadow_mask = white_img.astype(np.int32) - black_img.astype(np.int32)
    shadow_mask = np.greater(shadow_mask, parameters[1])
    shadow_mask = shadow_mask.astype(np.uint8) * 255
    gray_col_stack = np.stack(images[0:n_cols:2]).astype(np.int32) - np.stack(images[1:n_cols:2]).astype(np.int32)
    gray_col = np.greater(gray_col_stack, 0)
    dec_mask_col = np.greater(np.abs(gray_col_stack), parameters[0])
    dec_mask_col = np.logical_and.reduce(dec_mask_col, axis=0)
    gray_row_stack = np.stack(images[n_cols:n_cols+n_rows:2]).astype(np.int32) - np.stack(images[n_cols+1:n_cols+n_rows:2]).astype(np.int32)
    gray_row = np.greater(gray_row_stack, 0)
    dec_mask_row = np.greater(np.abs(gray_row_stack), parameters[0])
    dec_mask_row = np.logical_and.reduce(dec_mask_row, axis=0)
    dec_mask = np.logical_and(dec_mask_col, dec_mask_row)
    mask = np.logical_and(dec_mask, shadow_mask)
    p2s = {}
    for i in range(mask.shape[1]):
        for j in range(mask.shape[0]):
            if mask[j, i]:
                xDec = gray_decoder(gray_col[:, j, i])
                yDec = gray_decoder(gray_row[:, j, i])
                if xDec < pro_size[0] and yDec < pro_size[1]:
                    prj_idx = (parameters[2], xDec, yDec) # Projector number and matched coordinates
                    if prj_idx not in p2s:
                        p2s[prj_idx] = []
                    p2s[prj_idx].append((i, j))
    return p2s