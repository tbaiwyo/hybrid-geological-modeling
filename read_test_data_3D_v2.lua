function readData()
dataTrain = torch.FloatTensor(50,30,64,64)
local dataTable = {}
local file_name = 'Cate_3D_CCSIM_Result_Extend_HD.txt'
local f = io.open(file_name, r)
for i in f:lines() do
  table.insert(dataTable, i)
end
f:close()

for i = 1, 50 do
  for j = 1, 30 do
    for k = 1, 64 do
      for m = 1, 64 do
        dataTrain[{{i,i}, {j,j}, {k,k}, {m,m}}] = dataTable[(i - 1) * 30 * 64 * 64 + (j - 1) * 64 * 64 + (k - 1) * 64 + m]
      end
    end
  end
end
dataTable = nil
collectgarbage()
dataTrain = dataTrain * 80
return dataTrain
end