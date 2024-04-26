from dotmap import DotMap


a = DotMap()
a.f = 1
print(a.f)
print(a.seed == DotMap())