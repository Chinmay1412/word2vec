require 'nn'
require 'optim'

corpus = 'linescorpus.txt'
--tensortype = torch.getdefaulttensortype()
minfreq = 5
dim = 200
word = torch.IntTensor(1) 
window = 5 
lr = 0.07
vocab = {}
index2word = {}
word2index = {}
total_count = 0
batchsize = 200;  -- Mini-batch size.
total_windows = 0
fptr = nil
window_words=torch.IntTensor(window-1)
word_vecs_norm=nil
word_vecs=nil
-- This function trains a neural network language model.
function train(epochs,use_manual_technique)
--[[% Inputs:
%   epochs: Number of epochs to run.
% Output:
%   model: A struct containing the learned weights and biases and vocabulary.
]]--

start_time = os.clock();

-- SET HYPERPARAMETERS HERE.

learning_rate = lr;  -- Learning rate; default = 0.1.
momentum = 0.9;  -- Momentum; default = 0.9.
numhid1 = dim;  -- Dimensionality of embedding space; default = 50.
numhid2 = 300;  -- Number of units in hidden layer; default = 200.

-- LOAD DATA.
build_vocab()

numwords = window-1
numbatches = math.ceil((total_windows-window+1)/batchsize)

print(vocab_size)
print(total_windows)
print(numwords)
print(numbatches)

-- Create the neural net.
model = nn.Sequential();
word_vecs = nn.LookupTable(vocab_size, numhid1);
model:add( word_vecs ); -- lookuptable, so for 3 inputs (words) will produce a 3 x 50 matrix.
model:add( nn.Reshape(numwords*numhid1));        -- reshape 3 x 50 matrix to 150 units which is the first layer.
model:add( nn.Linear(numwords*numhid1,numhid2)); -- second layer is 200 units.
model:add( nn.Sigmoid() );                       -- activation function.
model:add( nn.Linear(numhid2,vocab_size) );      -- output layer is 250 units.
model:add( nn.LogSoftMax() ); 

-- Minimize the negative log-likelihood
criterion = nn.ClassNLLCriterion();
trainer = nn.StochasticGradient(model, criterion);
trainer.learningRate = learning_rate;
trainer.maxIteration = 1;


print(string.format("Total #%.0f batches",numbatches));

-- Train the model.
for epoch = 1,epochs do
    print(string.format('Epoch %.0f', epoch));
  
  	local foo = 1
	fptr = io.open(corpus, "r")
	for line in fptr:lines() do
	  word=line
	  word_idx = word2index[word]
	  if word_idx ~= nil then -- word exists in vocab
	    window_words[foo] = word_idx
	    if foo==window-1 then break end
	    foo = foo+1
	  end 
	end

  for m = 1, numbatches do
    print(string.format("Batch #%.0f",m));
    if use_manual_technique == false then
      
      dataset,_,_ = getNextBatch();
      trainer:train(dataset); 
      
    else 
      -- Manual Training.
      _,inputs,targets = getNextBatch();
      
      optimState = { 
        learningRate = learning_rate,
        momentum = momentum
      };
      parameters,gradParameters = model:getParameters();
      
      -- call the stochastic gradient optimizer.
      optim.sgd(feval, parameters, optimState) 
    end

  end
  fptr:close()
end

diff = os.clock() - start_time;
print(string.format('Training took %.2f seconds\n', diff));

model.vocab = word2index;
model.vocab_ByIndex = index2word;

dump_wordvector();

return model;

end

-- Returns the next batch as a single datasource.
function getNextBatch()
  
  input_batch=torch.IntTensor(batchsize,window-1)
  target_batch=torch.IntTensor(batchsize,1)

  print("Training...")
  local start = sys.clock()
  local cnt = 1
  p=((window-1)/2);
  p=p+1;
  for line in fptr:lines() do
      word=line
      word_idx = word2index[word]
      bar=p;
      if word_idx ~= nil then -- word exists in vocab
      	remword=window_words[bar]
      	for j=bar,window-2 do
      		window_words[j]=window_words[j+1];
      	end
      	window_words[window-1]=word_idx
        input_batch[cnt]=window_words
        --print(window_words)
        target_batch[cnt]=remword
        
        for j=1,bar-2 do
        	window_words[j]=window_words[j+1];
        end
        window_words[bar-1]=remword
        cnt=cnt+1
        if cnt==batchsize+1 then break end
      end
  end

  dataset = {}
  cnt=cnt-1
  function dataset:size() return cnt end
  for i=1,dataset:size() do
      local input = input_batch[i];  
      local output = target_batch[i];
          
      dataset[i] = {input, output}
  end

  return dataset, input_batch, target_batch;
end  

-- Create a “closure” feval(x) that takes the 
-- parameter vector as argument and returns 
-- the loss and its gradient on the batch.
-- This function will be used by the optimizer (optim.sgd).
feval = function(x)
  -- get new parameters
  parameters:copy(x)

  -- reset gradients
  gradParameters:zero()

  -- f is the average of all criterions
  local f = 0

  -- evaluate function for complete mini batch
  for i = 1,(#inputs)[1] do
    -- estimate f
    local output = model:forward(inputs[i])
    local err = criterion:forward(output, targets[i])
    f = f + err

    -- estimate df/dW
    local df_do = criterion:backward(output, targets[i])
    model:backward(inputs[i], df_do) -- backprop.
  end      

  -- normalize gradients and f(X)
  gradParameters:div((#inputs)[1])
  f = f/(#inputs)[1];

  print(string.format("error: %f", f));
  -- return f and df/dX
  return f,gradParameters 
end

-- Build vocab frequency, word2index, and index2word from input file
function build_vocab()
    print("Building vocabulary...")
    local start = sys.clock()
    local f = io.open(corpus, "r")
    local n = 1
    for line in f:lines() do
      word=line
      total_count = total_count + 1
      if vocab[word] == nil then
        vocab[word] = 1   
      else
        vocab[word] = vocab[word] + 1
      end
      n = n + 1
    end
    f:close()
    -- Delete words that do not meet the minfreq threshold and create word indices
    for word, count in pairs(vocab) do
      if count >= minfreq then
          index2word[#index2word+1] = word
          word2index[word] = #index2word
          total_windows = total_windows + count   
      else
        vocab[word] = nil
      end
    end
    vocab_size = #index2word
    print(string.format("%d words and %d sentences processed in %.2f seconds.", total_count, n, sys.clock() - start))
    print(string.format("Vocab size after eliminating words occuring less than %d times: %d", minfreq, vocab_size))
end

-- split on separator
function split(input, sep)
    if sep == nil then
        sep = "%s"
    end
    local t = {}; local i = 1
    for str in string.gmatch(input, "([^"..sep.."]+)") do
        t[i] = str; i = i + 1
    end
    return t
end

-- Row-normalize a matrix
function normalize(m)
    m_norm = torch.zeros(m:size())
    for i = 1, m:size(1) do
    	m_norm[i] = m[i] / torch.norm(m[i])
    end
    return m_norm
end

-- function to write word vector into file
function dump_wordvector()
	if word_vecs_norm == nil then
        word_vecs_norm = normalize(word_vecs.weight:double())
    end

    local out = io.open("wordvector.txt", "wb")
    for idx = 1, vocab_size do
    	out:write(index2word[idx])
    	for i = 1, dim do
    		out:write(" ",string.format("%.6f",word_vecs_norm[idx][i]))
    	end
    	out:write("\n")
    end
    out:close()
end