# OnceNAS: Discovering Efficient On-device Inference Neural Networks for Edge Devices

To address the scarcity of benchmark datasets in the field of lightweight NAS, we introduce the DARTS-Bench dataset. DARTS-Bench includes candidate architectures sampled from the DARTS search space, which have been trained to converge to provide precise accuracy measurements. In addition, we collect corresponding runtime information on various hardware platforms and structure it for easy access. We also provide API functions for convenient data retrieval, which will greatly facilitate future research efforts. Most existing lightweight NAS methods are based on DARTS search space, which we also adopt as our testing benchmark.

# Usage:

**Requirements**
```
from api_darts import DartsBenchAPI as API
api = API('data_545.pkl', verbose=False)
```

**Search based on network architecture**
```
str = "Genotype(normal=[('max_pool_3x3', 0), ('sep_conv_5x5', 1), ('sep_conv_5x5', 0), ('dil_conv_5x5', 2), ('avg_pool_3x3', 2), ('dil_conv_3x3', 1), ('dil_conv_3x3', 2), ('none', 0)], normal_concat=range(2, 6), reduce=[('dil_conv_3x3', 0), ('none', 1), ('sep_conv_5x5', 2), ('avg_pool_3x3', 0), ('skip_connect', 0), ('dil_conv_3x3', 1), ('dil_conv_5x5', 0), ('sep_conv_5x5', 3)], reduce_concat=range(2, 6))"

arch = api.str2lists( str, reduction=True )
print ('there are {:} nodes with optional operations in this reduce cell'.format(len(arch)))
for i, node in enumerate(arch):
    print('the {:}-th node (node {:}) is with op: {:}'.format(i+3,i+2,node))

arch = api.str2lists( str, reduction=False )
print ('there are {:} nodes with optional operations in this normal cell'.format(len(arch)))
for i, node in enumerate(arch):
    print('the {:}-th node (node {:}) is with op: {:}'.format(i+3,i+2,node))

arch = api.str2matrix( str )
print(arch)
```

**Search based on parameters**
```
suit_list = api.find_suit_list(value, acc, late, flop, param)
```"# OnceNAS_Bench" 
