def batchify(X, n):
  l = len(X)
  for ndx in range(0, l, n):
    x = X[ndx:min(ndx + n, l)]
    if len(x) == n:
      yield x

def write_pcd(path, points):
  num_points = len(points)
  info_text = "# .PCD v0.7 - Point Cloud Data file format\nVERSION 0.7\nFIELDS x y z\nSIZE 4 4 4\nTYPE F F F\nCOUNT 1 1 1\nWIDTH {}\nHEIGHT 1\nVIEWPOINT 0 0 0 1 0 0 0\nPOINTS {}\nDATA ascii\n".format(
    num_points, num_points)
  with open(path, "w+") as f:
    f.write(info_text)
    for p in points:
      f.write('{} {} {}\n'.format(p[0], p[1], p[2]))