
class My_Random:
    def __init__(self):
        self.seed = 0
    
    def set_seed(self, seed):
        self.seed = seed

    def dice(self):
        sq = str(self.seed**2)
        sq_8 = sq.zfill(8)
        middle = int(sq_8[3:7])
        return (middle%6)+1

my_rand = My_Random()
my_rand.set_seed(1234)
print(my_rand.dice())  # First roll
print(my_rand.dice())  # Second roll
print(my_rand.dice())