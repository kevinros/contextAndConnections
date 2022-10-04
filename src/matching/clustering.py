import re

class disjoint_set:
    def __init__(self, n):
        self.vals = [-1 for i in range(n)]
    def query(self, n):
        if self.vals[n] < 0:
            return n
        q = self.query(self.vals[n])
        self.vals[n] = q
        return q
    def merge(self, n1, n2):
        q1 = self.query(n1)
        q2 = self.query(n2)
        if (q1 == q2):
            return
        if (self.vals[q1] < self.vals[q2]):
            #q1 is larger than q2
            self.vals[q1] += self.vals[q2]
            self.vals[q2] = q1
        else:
            self.vals[q2] += self.vals[q1]
            self.vals[q1] = q2
    def find_set_size(self, n):
        return self.vals[self.query(n)]
    def count_nonzero(self):
        count = 0
        for i in self.vals:
            if i < 0:
                count += 1
        return count
    def largest_set(self):
        return -min(self.vals)
      
parser = re.compile("\((\d+), (\d+)\): (\d\.\d*)")
def cluster(matches_file, threshold, n_files):
  ds = disjoint_set(n_files)
  with open(matches_file, "r") as f:
    while True:
        line = f.readline()
        if line is None or line == "":
            break
        i1, i2, val = parser.match(line).groups()
        i1 = int(i1)
        i2 = int(i2)
        val = float(val)
        if val > threshold:
            ds.merge(i1, i2)
  return ds

def write_out(ds, files):
  with open("clusters.csv", "w+") as f:
      f.write("file, cluster\n")
      for i, im in enumerate(ds.vals):
          if im < 0:
              f.write(files[i] + ", " + files[i] +"\n")
          else:
              f.write(files[i] + ", " + ds.query(files[im]) +"\n")
