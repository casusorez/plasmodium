
def split_image(img, window_size, margin):
    sh = list(img.shape)
    sh[0], sh[1] = sh[0] + margin * 2, sh[1] + margin * 2
    img_ = np.zeros(shape=sh)
    img_[margin:-margin, margin:-margin] = img
    stride = window_size
    step = window_size + 2 * margin
    nrows, ncols = img.shape[0] // window_size, img.shape[1] // window_size
    splitted = []
    final_image = np.zeros_like(img)
    # on splitte l'image
    for i in range(nrows):
        for j in range(ncols):
            h_start = j * stride
            v_start = i * stride
            cropped = img_[v_start:v_start + step, h_start:h_start + step]
            splitted.append(cropped.astype('uint8'))

    return splitted