require 'train'
--require 'predict_next_word'

use_manual_technique = false;
epochs = 2;

-- Manual Training seems to require more epochs to get a similar error rate.
if use_manual_technique == true then epochs = 2; end

model = train(epochs,use_manual_technique);
