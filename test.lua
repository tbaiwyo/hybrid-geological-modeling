require 'nn'
require 'torch'
require 'image'
require 'read_test_data_3D_v2'
util = paths.dofile('util.lua')
torch.setdefaulttensortype('torch.FloatTensor')


opt = {
  batchSize = 50,       
  net = 'checkpoints/Cate_Cond_3D_400_net_G.t7',    
  name = 'Cate_Cond_3D',        
  gpu = 1,               
  training_nc = 50,
  nc = 30,              
  display = 0,           
  loadSize = 150,         
  fineSize = 64,        
  nThreads = 1,          
  manualSeed = 0,        
  overlapPred = 0,       

  noiseGen = 0,          
  noisetype = 'normal',  
  nz = 4000,             
}
for k,v in pairs(opt) do opt[k] = tonumber(os.getenv(k)) or os.getenv(k) or opt[k] end
print(opt)
if opt.display == 0 then opt.display = false end
if opt.noiseGen == 0 then opt.noiseGen = false end

if opt.manualSeed == 0 then
    opt.manualSeed = torch.random(1, 10000)
end
print("Seed: " .. opt.manualSeed)
torch.manualSeed(opt.manualSeed)

assert(opt.net ~= '', 'provide a generator model')
net = util.load(opt.net, opt.gpu)
net:evaluate()

input_image_ctx = torch.Tensor(opt.batchSize, opt.nc, opt.fineSize, opt.fineSize)
local noise 
if opt.noiseGen then
    noise = torch.Tensor(opt.batchSize, opt.nz, 1, 1)
    if opt.noisetype == 'uniform' then
        noise:uniform(-1, 1)
    elseif opt.noisetype == 'normal' then
        noise:normal(0, 1)
    end
end

if opt.gpu > 0 then
    require 'cunn'
    if pcall(require, 'cudnn') then
        print('Using CUDNN !')
        require 'cudnn'
        net = util.cudnn(net)
    end
    net:cuda()
    input_image_ctx = input_image_ctx:cuda()
    if opt.noiseGen then
        noise = noise:cuda()
    end
else
    net:float()
end
print(net)

local data = readData() 
print("Dataset Size: ", data:size())

local image_ctx = torch.Tensor(opt.batchSize, opt.nc, opt.fineSize, opt.fineSize)

image_ctx[{{},{},{},{}}] = data[{{},{},{},{}}]

image_ctx = 2*image_ctx/255.0 - 1.0

print('Loaded Image Block: ', image_ctx:size(1)..' x '..image_ctx:size(2) ..' x '..image_ctx:size(3)..' x '..image_ctx:size(4))

real_center = image_ctx[{{},{},{17,48},{17,48}}]:clone() 

input_image_ctx:copy(image_ctx)
input_image_ctx[{{},{},{17, 21},{17, 48}}] = 2*64.8/255.0 - 1.0
input_image_ctx[{{},{},{22, 22},{17, 21}}] = 2*64.8/255.0 - 1.0
input_image_ctx[{{},{},{22, 22},{23, 48}}] = 2*64.8/255.0 - 1.0
input_image_ctx[{{},{},{23, 42},{17, 48}}] = 2*64.8/255.0 - 1.0
input_image_ctx[{{},{},{43, 43},{17, 42}}] = 2*64.8/255.0 - 1.0
input_image_ctx[{{},{},{43, 43},{44, 48}}] = 2*64.8/255.0 - 1.0
input_image_ctx[{{},{},{44, 48},{17, 48}}] = 2*64.8/255.0 - 1.0

local input_image = torch.Tensor(opt.batchSize, opt.nc, opt.fineSize, opt.fineSize)
input_image:copy(input_image_ctx)

local pred_center
if opt.noiseGen then
    pred_center = net:forward({input_image_ctx,noise})
else
    pred_center = net:forward(input_image_ctx)
end
print('Prediction: size: ', pred_center:size(1)..' x '..pred_center:size(2) ..' x '..pred_center:size(3)..' x '..pred_center:size(4))
print('Prediction: Min, Max, Mean, Stdv: ', pred_center:min(), pred_center:max(), pred_center:mean(), pred_center:std())

input_image_ctx[{{},{},{17,48},{17,48}}]:copy(pred_center[{{},{},{1,32},{1,32}}])

input_image:add(1):mul(0.5)  
input_image_ctx:add(1):mul(0.5)  
image_ctx:add(1):mul(0.5)  
input_image_ctx = torch.round(input_image_ctx / 0.3137)

if opt.display then
    disp = require 'display'
    disp.image(pred_center, {win=1000, title=opt.name})
    disp.image(real_center, {win=1001, title=opt.name})
    disp.image(image_ctx, {win=1002, title=opt.name})
    print('Displayed image in browser !')
end

local Out_now = false
if Out_now then
  gFile = io.open('3D_Cate_Test_Result_400.txt','w')

  for i = 1,50 do
    for j = 1,30 do
      for k = 1,64 do
        for l = 1,64 do
          gFile:write(tostring(input_image_ctx[i][j][k][l]))
          gFile:write('\n')
        end
      end
    end
  end
    
  gFile:close()
end
