require 'torch'
require 'nn'
require 'optim'
require 'image'
require 'read_data_3D_v2.lua'
util = paths.dofile('util.lua')

opt = {
   batchSize = 64,         
   loadSize = 150,         
   fineSize = 64,         
   nBottleneck = 4000,      
   nef = 64,               
   ngf = 64,               
   ndf = 64,               
   training_nc = 50,
   nc = 30,                 
   wtl2 = 0.999,               
   overlapPred = 0,        
   nThreads = 4,           
   niter = 1000,             
   lr = 0.0001,            
   beta1 = 0.5,            
   ntrain = 1920,    
   nfile = 5,             
   display = 0,           
   display_id = 10,        
   display_iter = 50,      
   gpu = 1,                
   name = 'Cate_Cond_3D_Test',      
   manualSeed = 0,         
   conditionAdv = 0,      
   noiseGen = 0,          
   noisetype = 'normal',   
   nz = 4000,              
}
for k,v in pairs(opt) do opt[k] = tonumber(os.getenv(k)) or os.getenv(k) or opt[k] end
print(opt)
if opt.display == 0 then opt.display = false end
if opt.conditionAdv == 0 then opt.conditionAdv = false end
if opt.noiseGen == 0 then opt.noiseGen = false end

if opt.manualSeed == 0 then
    opt.manualSeed = torch.random(1, 10000)
end
print("Seed: " .. opt.manualSeed)
torch.manualSeed(opt.manualSeed)
torch.setnumthreads(1)
torch.setdefaulttensortype('torch.FloatTensor')

---------------------------------------------------------------------------
local function weights_init(m)
   local name = torch.type(m)
   if name:find('Convolution') then
      m.weight:normal(0.0, 0.02)
      m.bias:fill(0)
   elseif name:find('BatchNormalization') then
      if m.weight then m.weight:normal(1.0, 0.02) end
      if m.bias then m.bias:fill(0) end
   end
end

local nc = opt.nc
local nz = opt.nz
local nBottleneck = opt.nBottleneck
local ndf = opt.ndf
local ngf = opt.ngf
local nef = opt.nef
local real_label = 1
local fake_label = 0

local SpatialBatchNormalization = nn.SpatialBatchNormalization
local SpatialConvolution = nn.SpatialConvolution
local SpatialFullConvolution = nn.SpatialFullConvolution

---------------------------------------------------------------------------
local netE = nn.Sequential()
netE:add(SpatialConvolution(nc, nef, 4, 4, 2, 2, 1, 1))
netE:add(nn.LeakyReLU(0.2, true))

netE:add(SpatialConvolution(nef, nef * 2, 4, 4, 2, 2, 1, 1))
netE:add(SpatialBatchNormalization(nef * 2)):add(nn.LeakyReLU(0.2, true))

netE:add(SpatialConvolution(nef * 2, nef * 4, 4, 4, 2, 2, 1, 1))
netE:add(SpatialBatchNormalization(nef * 4)):add(nn.LeakyReLU(0.2, true))

netE:add(SpatialConvolution(nef * 4, nef * 8, 4, 4, 2, 2, 1, 1))
netE:add(SpatialBatchNormalization(nef * 8)):add(nn.LeakyReLU(0.2, true))

netE:add(SpatialConvolution(nef * 8, nBottleneck, 4, 4))

local netG = nn.Sequential()
local nz_size = nBottleneck
if opt.noiseGen then
    local netG_noise = nn.Sequential()
    netG_noise:add(SpatialConvolution(nz, nz, 1, 1, 1, 1, 0, 0))

    local netG_pl = nn.ParallelTable();
    netG_pl:add(netE)
    netG_pl:add(netG_noise)

    netG:add(netG_pl)
    netG:add(nn.JoinTable(2))
    netG:add(SpatialBatchNormalization(nBottleneck+nz)):add(nn.LeakyReLU(0.2, true))

    nz_size = nBottleneck+nz
else
    netG:add(netE)
    netG:add(SpatialBatchNormalization(nBottleneck)):add(nn.LeakyReLU(0.2, true))

    nz_size = nBottleneck
end

netG:add(SpatialFullConvolution(nz_size, ngf * 8, 4, 4))
netG:add(SpatialBatchNormalization(ngf * 8)):add(nn.ReLU(true))

netG:add(SpatialFullConvolution(ngf * 8, ngf * 4, 4, 4, 2, 2, 1, 1))
netG:add(SpatialBatchNormalization(ngf * 4)):add(nn.ReLU(true))

netG:add(SpatialFullConvolution(ngf * 4, ngf * 2, 4, 4, 2, 2, 1, 1))
netG:add(SpatialBatchNormalization(ngf * 2)):add(nn.ReLU(true))

netG:add(SpatialFullConvolution(ngf * 2, nc, 4, 4, 2, 2, 1, 1))
netG:add(nn.Tanh())

netG:apply(weights_init)

---------------------------------------------------------------------------
local netD = nn.Sequential()
if opt.conditionAdv then
    local netD_ctx = nn.Sequential()
    netD_ctx:add(SpatialConvolution(nc, ndf, 5, 5, 2, 2, 2, 2))

    local netD_pred = nn.Sequential()
    netD_pred:add(SpatialConvolution(nc, ndf, 5, 5, 2, 2, 2+32, 2+32))      

    local netD_pl = nn.ParallelTable();
    netD_pl:add(netD_ctx)
    netD_pl:add(netD_pred)

    netD:add(netD_pl)
    netD:add(nn.JoinTable(2))
    netD:add(nn.LeakyReLU(0.2, true))

    netD:add(SpatialConvolution(ndf*2, ndf, 4, 4, 2, 2, 1, 1))
    netD:add(SpatialBatchNormalization(ndf)):add(nn.LeakyReLU(0.2, true))
else
    netD:add(SpatialConvolution(nc, ndf, 4, 4, 2, 2, 1, 1))
    netD:add(nn.LeakyReLU(0.2, true))
end
netD:add(SpatialConvolution(ndf, ndf * 2, 4, 4, 2, 2, 1, 1))
netD:add(SpatialBatchNormalization(ndf * 2)):add(nn.LeakyReLU(0.2, true))
netD:add(SpatialConvolution(ndf * 2, ndf * 4, 4, 4, 2, 2, 1, 1))
netD:add(SpatialBatchNormalization(ndf * 4)):add(nn.LeakyReLU(0.2, true))

netD:add(SpatialConvolution(ndf * 4, 1, 4, 4))
netD:add(nn.Sigmoid())
netD:add(nn.View(1):setNumInputDims(3))

netD:apply(weights_init)

---------------------------------------------------------------------------
local criterion = nn.BCECriterion()
local criterionMSE
if opt.wtl2~=0 then
  criterionMSE = nn.MSECriterion()
end

---------------------------------------------------------------------------
print('LR of Gen is ',(opt.wtl2>0 and opt.wtl2<1) and 10 or 1,'times Adv')
optimStateG = {
   learningRate = (opt.wtl2>0 and opt.wtl2<1) and opt.lr*10 or opt.lr,
   beta1 = opt.beta1,
}
optimStateD = {
   learningRate = opt.lr,
   beta1 = opt.beta1,
}

---------------------------------------------------------------------------
local input_ctx_vis = torch.Tensor(opt.batchSize, nc, opt.fineSize, opt.fineSize)
local input_ctx = torch.Tensor(opt.batchSize, nc, opt.fineSize, opt.fineSize)
local input_center = torch.Tensor(opt.batchSize, nc, opt.fineSize/2, opt.fineSize/2)
local data = torch.Tensor(opt.nfile, opt.training_nc, opt.loadSize, opt.loadSize) 
local real_ctx = torch.Tensor(opt.batchSize, nc, opt.fineSize, opt.fineSize) 
local real_center = torch.Tensor(opt.batchSize, nc, opt.fineSize/2, opt.fineSize/2)
local input_real_center
if opt.wtl2~=0 then
    input_real_center = torch.Tensor(opt.batchSize, nc, opt.fineSize/2, opt.fineSize/2)
end
local noise = torch.Tensor(opt.batchSize, nz, 1, 1)  
local label = torch.Tensor(opt.batchSize)
local errD, errG, errG_l2
local epoch_tm = torch.Timer()
local tm = torch.Timer()

if pcall(require, 'cudnn') and pcall(require, 'cunn') and opt.gpu>0 then
    print('Using CUDNN !')
end
if opt.gpu > 0 then
    require 'cunn'
    cutorch.setDevice(opt.gpu)
    input_ctx_vis = input_ctx_vis:cuda(); input_ctx = input_ctx:cuda();  input_center = input_center:cuda()
    data = data:cuda(); real_ctx = real_ctx:cuda();real_center = real_center:cuda(); 
    -------
    noise = noise:cuda();  label = label:cuda()
    netG = util.cudnn(netG);     netD = util.cudnn(netD)
    netD:cuda();           netG:cuda();           criterion:cuda();      
    if opt.wtl2~=0 then
      criterionMSE:cuda(); input_real_center = input_real_center:cuda();
    end
end

print('NetG:',netG)
print('NetD:',netD)

local parametersD, gradParametersD = netD:getParameters()
local parametersG, gradParametersG = netG:getParameters()

if opt.display then disp = require 'display' end

noise_vis = noise:clone()
if opt.noisetype == 'uniform' then
    noise_vis:uniform(-1, 1)
elseif opt.noisetype == 'normal' then
    noise_vis:normal(0, 1)
end

---------------------------------------------------------------------------
local fDx = function(x)
   netD:apply(function(m) if torch.type(m):find('Convolution') then m.bias:zero() end end)
   netG:apply(function(m) if torch.type(m):find('Convolution') then m.bias:zero() end end)
   gradParametersD:zero()
   data = readData() 
   
   local j
   local k
   local l
   local m 
   for i = 1, opt.batchSize do
     j = torch.random(1, opt.nfile)
     m = torch.random(1, opt.training_nc - opt.nc + 1)
     k = torch.random(1, opt.loadSize - opt.fineSize + 1)
     l = torch.random(1, opt.loadSize - opt.fineSize + 1)
     real_ctx[{{i, i},{},{},{}}] = data[{{j, j},{m, m + opt.nc - 1},{k, k + opt.fineSize - 1},{l, l + opt.fineSize - 1}}]
   end
   data = nil
   collectgarbage()
   real_ctx = 2*real_ctx/255.0 - 1.0   
   
   real_center = real_ctx[{{},{},{17,48},{17,48}}]:clone() 
   
   real_ctx[{{},{},{17,21},{17,48}}] = 2 * 64.8 / 255.0 - 1.0
   
   real_ctx[{{},{},{22,22},{17,21}}] = 2 * 64.8 / 255.0 - 1.0
   real_ctx[{{},{},{22,22},{23,48}}] = 2 * 64.8 / 255.0 - 1.0
   
   real_ctx[{{},{},{23,42},{17,48}}] = 2 * 64.8 / 255.0 - 1.0
   
   real_ctx[{{},{},{43,43},{17,42}}] = 2 * 64.8 / 255.0 - 1.0
   real_ctx[{{},{},{43,43},{44,48}}] = 2 * 64.8 / 255.0 - 1.0
   
   real_ctx[{{},{},{44,48},{17,48}}] = 2 * 64.8 / 255.0 - 1.0
   
   input_ctx:copy(real_ctx)
   input_center:copy(real_center)
   if opt.wtl2~=0 then
      input_real_center:copy(real_center)
   end
   label:fill(real_label)

   local output
   if opt.conditionAdv then
      output = netD:forward({input_ctx,input_center})
   else
      output = netD:forward(input_center)
   end
   local errD_real = criterion:forward(output, label)
   local df_do = criterion:backward(output, label)
   if opt.conditionAdv then
      netD:backward({input_ctx,input_center}, df_do)
   else
      netD:backward(input_center, df_do)
   end
   
   if opt.noisetype == 'uniform' then 
       noise:uniform(-1, 1)
   elseif opt.noisetype == 'normal' then
       noise:normal(0, 1)
   end
   local fake
   if opt.noiseGen then
      fake = netG:forward({input_ctx,noise})
   else
      fake = netG:forward(input_ctx)
   end
   input_center:copy(fake)
   label:fill(fake_label)

   local output
   if opt.conditionAdv then
      output = netD:forward({input_ctx,input_center})
   else
      output = netD:forward(input_center)
   end
   local errD_fake = criterion:forward(output, label)
   local df_do = criterion:backward(output, label)
   if opt.conditionAdv then
      netD:backward({input_ctx,input_center}, df_do)
   else
      netD:backward(input_center, df_do)
   end

   errD = errD_real + errD_fake

   return errD, gradParametersD
end


local fGx = function(x)
   netD:apply(function(m) if torch.type(m):find('Convolution') then m.bias:zero() end end)
   netG:apply(function(m) if torch.type(m):find('Convolution') then m.bias:zero() end end)
   gradParametersG:zero()

   label:fill(real_label) 

   local output = netD.output 
   errG = criterion:forward(output, label)
   local df_do = criterion:backward(output, label)
   local df_dg
   if opt.conditionAdv then
      df_dg = netD:updateGradInput({input_ctx,input_center}, df_do)
      df_dg = df_dg[2]    
   else
      df_dg = netD:updateGradInput(input_center, df_do)
   end

   local errG_total = errG
   if opt.wtl2~=0 then
      errG_l2 = criterionMSE:forward(input_center, input_real_center)
      local df_dg_l2 = criterionMSE:backward(input_center, input_real_center)

      if opt.overlapPred==0 then
        if (opt.wtl2>0 and opt.wtl2<1) then
          df_dg:mul(1-opt.wtl2):add(opt.wtl2,df_dg_l2)
          errG_total = (1-opt.wtl2)*errG + opt.wtl2*errG_l2
        else
          df_dg:add(opt.wtl2,df_dg_l2)
          errG_total = errG + opt.wtl2*errG_l2
        end
      else
        local overlapL2Weight = 10
        local wtl2Matrix = df_dg_l2:clone():fill(overlapL2Weight*opt.wtl2)
        wtl2Matrix[{{},{},{1 + opt.overlapPred, opt.fineSize/2 - opt.overlapPred},{1 + opt.overlapPred, opt.fineSize/2 - opt.overlapPred}}]:fill(opt.wtl2)
        if (opt.wtl2>0 and opt.wtl2<1) then
          df_dg:mul(1-opt.wtl2):addcmul(1,wtl2Matrix,df_dg_l2)
          errG_total = (1-opt.wtl2)*errG + opt.wtl2*errG_l2
        else
          df_dg:addcmul(1,wtl2Matrix,df_dg_l2)
          errG_total = errG + opt.wtl2*errG_l2
        end
      end
   end

   if opt.noiseGen then
      netG:backward({input_ctx,noise}, df_dg)
   else
      netG:backward(input_ctx, df_dg)
   end

   return errG_total, gradParametersG
end

---------------------------------------------------------------------------
for epoch = 1, opt.niter do
   epoch_tm:reset()
   local counter = 0
   for i = 1, opt.batchSize * 30, opt.batchSize do
      tm:reset()
      optim.adam(fDx, parametersD, optimStateD)
      optim.adam(fGx, parametersG, optimStateG)
      
      counter = counter + 1
      if counter % opt.display_iter == 0 and opt.display then
          local real_ctx = data:getBatch()
          local real_center = real_ctx[{{},{},{1,32},{97,128}}]:clone()
          
          real_ctx[{{},{},{1, 32},{97, 128}}] = 2*128.0/255.0 - 1.0
          
          input_ctx_vis:copy(real_ctx)
          local fake
          if opt.noiseGen then
            fake = netG:forward({input_ctx_vis,noise_vis})
          else
            fake = netG:forward(input_ctx_vis)
          end
          
          real_ctx[{{},{},{1,32},{97,128}}]:copy(fake[{{},{},{1,32},{1,32}}])          
          
          disp.image(fake, {win=opt.display_id, title=opt.name})
          disp.image(real_center, {win=opt.display_id * 3, title=opt.name})
          disp.image(real_ctx, {win=opt.display_id * 6, title=opt.name})
      end

      if ((i-1) / opt.batchSize) % 1 == 0 then
         print(('Epoch: [%d][%8d / %8d]\t Time: %.3f  '
                   .. '  Err_G_L2: %.4f   Err_G: %.4f  Err_D: %.4f'):format(
                 epoch, ((i-1) / opt.batchSize),
                 opt.ntrain / opt.batchSize,
                 tm:time().real, errG_l2 or -1,
                 errG and errG or -1, errD and errD or -1))
      end
   end
   paths.mkdir('checkpoints')
   parametersD, gradParametersD = nil, nil 
   parametersG, gradParametersG = nil, nil
   if epoch % 10 == 0 then
      util.save('checkpoints/' .. opt.name .. '_' .. epoch .. '_net_G.t7', netG, opt.gpu)
      util.save('checkpoints/' .. opt.name .. '_' .. epoch .. '_net_D.t7', netD, opt.gpu)
   end
   parametersD, gradParametersD = netD:getParameters() 
   parametersG, gradParametersG = netG:getParameters()
   print(('End of epoch %d / %d \t Time Taken: %.3f'):format(
            epoch, opt.niter, epoch_tm:time().real))
end
