local util = {}

local function recursiveTableClear(mAttribute, compulsoryAttr)
    compulsoryAttr = compulsoryAttr==nil and true or compulsoryAttr
    if type(mAttribute) == 'table' then
        for i=1,#mAttribute do
            mAttribute[i] = recursiveTableClear(mAttribute[i])
        end
        return mAttribute
    else
        if compulsoryAttr then
            return mAttribute.new()
        else
            return mAttribute and mAttribute.new() or nil
        end
    end
end

local function recursiveSave(netsave)
    for k, l in ipairs(netsave.modules) do
        if netsave.modules[k].modules~=nil then
            recursiveSave(netsave.modules[k])
        end

        if torch.type(l) == 'cudnn.SpatialConvolution' then
            local new = nn.SpatialConvolution(l.nInputPlane, l.nOutputPlane,
                          l.kW, l.kH, l.dW, l.dH, 
                          l.padW, l.padH)
            new.weight:copy(l.weight)
            new.bias:copy(l.bias)
            netsave.modules[k] = new
        elseif torch.type(l) == 'fbnn.SpatialBatchNormalization' then
            new = nn.SpatialBatchNormalization(l.weight:size(1), l.eps, 
                           l.momentum, l.affine)
            new.running_mean:copy(l.running_mean)
            new.running_std:copy(l.running_std)
            if l.affine then
                new.weight:copy(l.weight)
                new.bias:copy(l.bias)
            end
            netsave.modules[k] = new
        endatchSize = 50,      
      net = 'checkpoints/Cate_Cond_3D_'..tostring(p*10)..'_net_G.t7',    
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

        local m = netsave.modules[k]
        m.output = recursiveTableClear(m.output)
        m.gradInput = recursiveTableClear(m.gradInput)
        m.finput = recursiveTableClear(m.finput,false)
        m.fgradInput = recursiveTableClear(m.fgradInput,false)
        m.buffer = nil
        m.buffer2 = nil
        m.centered = nil
        m.std = nil
        m.normalized = nil
        if m.weight then
            m.weight = m.weight:clone()
            m.gradWeight = m.gradWeight:clone()
            m.bias = m.bias:clone()
            m.gradBias = m.gradBias:clone()
        end
    end
end
function util.save(filename, net, gpu)
    net:float() 
    local netsave = net:clone()
    if gpu > 0 then
        net:cuda()
    end
    recursiveSave(netsave)
    netsave.output = recursiveTableClear(netsave.output)
    netsave.gradInput = recursiveTableClear(netsave.gradInput)

    netsave:apply(function(m) if m.weight then m.gradWeight = nil; m.gradBias = nil; end end)

    torch.save(filename, netsave)
end

function util.load(filename, gpu)
   local net = torch.load(filename)
   net:apply(function(m) if m.weight then 
        m.gradWeight = m.weight:clone():zero();
        m.gradBias = m.bias:clone():zero(); end end)
   return net
end

local function recursiveCudnn(net)
    for k, l in ipairs(net.modules) do
        if net.modules[k].modules~=nil then
            recursiveCudnn(net.modules[k])
        end
atchSize = 50,      
      net = 'checkpoints/Cate_Cond_3D_'..tostring(p*10)..'_net_G.t7',    
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
        if torch.type(l) == 'nn.SpatialConvolution' and pcall(require, 'cudnn') then
            local new = cudnn.SpatialConvolution(l.nInputPlane, l.nOutputPlane,
                         l.kW, l.kH, l.dW, l.dH,
                         l.padW, l.padH)
            new.weight:copy(l.weight)
            new.bias:copy(l.bias)
            net.modules[k] = new
        end
    end
end
function util.cudnn(net)
    recursiveCudnn(net)
    return net
end

return util
