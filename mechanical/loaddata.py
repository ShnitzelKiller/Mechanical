import onshape.brepio as brepio

loader = brepio.Loader('/projects/grail/benjones/cadlab/data')

did = '03180acbded3ea455c3e63ba'
mv = 'a5cc038a1c26099430d7314c'
eid = 'ee7cd5d7207130248b7845a2'

data = loader.load(did, mv, eid)
print(data)