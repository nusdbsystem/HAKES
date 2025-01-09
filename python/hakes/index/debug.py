def matrix_to_str(m):
    matrix_str = f"matrix: {m.shape}\n"
    if len(m.shape) > 2:
        return matrix_str
    if len(m.shape) == 1:
        matrix_str += f'row: {" ".join([f"{x:8.4f}" for x in m])}\n'
        return matrix_str
    if m.shape[1] == 1:
        matrix_str += f'row: {" ".join([f"{x[0]:8.4f}" for x in m])}\n'
        return matrix_str
    for i in range(m.shape[0]):
        matrix_str += f'row {i:04d}: {" ".join([f"{x:8.4f}" for x in m[i]])}\n'
    return matrix_str


def print_matrix(m):
    print(matrix_to_str(m))


def centroids_to_str(centroids):
    nlist, d = centroids.shape
    centroid_str = f"centroids: {centroids.shape}\n"
    for i in range(nlist):
        centroid_str += f'c{i:04d}: {" ".join([f"{x:8.4f}" for x in centroids[i]])}\n'
    return centroid_str


def print_centroids(centroids):
    print(centroids_to_str(centroids))


def codebook_to_str(codebook):
    m, ksub, dsub = codebook.shape
    code_str = f"codebook: {codebook.shape}\n"
    for j in range(ksub):
        code_str += f"code{j:02d}: |"
        for i in range(m):
            code_str += (
                f' {" ".join([f"{codebook[i, j, k]:8.4f}" for k in range(dsub)])} |'
            )
        code_str += "\n"
    return code_str


def print_codebook(codebook):
    print(codebook_to_str(codebook))
