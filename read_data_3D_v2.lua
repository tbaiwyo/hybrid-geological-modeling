function readData()
dataTrain = torch.FloatTensor(5,50,150,150)
local number_of_data_files = 40
local dataTable = {}
local random_number = torch.random(1, number_of_data_files)
print('file_index: '..random_number)
local prefix = 'Data_3D_Cate_DL/3D_cate_DL_'
local file_name = prefix..random_number..'.txt'
local f = io.open(file_name, r)
for i in f:lines() do
  table.insert(dataTable, i)
end
f:close()

for i = 1, 5 do
  for j = 1, 50 do
    for k = 1, 150 do
      for m = 1, 150 do
        dataTrain[{{i,i}, {j,j}, {k,k}, {m,m}}] = dataTable[(i - 1) * 50 * 150 * 150 + (j - 1) * 150 * 150 + (k - 1) * 150 + m]
      end
    end
  end
end
dataTable = nil
collectgarbage()
dataTrain = dataTrain * 80
return dataTrain
end