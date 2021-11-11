
import torch
import numpy as np


def rescale_two(x):
    
    """Rescales image input to be in range [-1,+1]."""
    
    print(x[x>0].shape)
    
    current_min = np.min(x)
    current_max = np.max(x)
    
    # we add an epsilon value to prevent division by zero
    epsilon = 1e-5
    rescaled_x = -1+((x - current_min)/np.max([current_max - current_min , epsilon]))*2
    
    print("After Rescaling:", rescaled_x[rescaled_x>0].shape)
    return rescaled_x


def rescale_input(x):
    
    """Rescales image input to be in range [0,1]."""
    
    current_min = np.min(x)
    current_max = np.max(x)
    
    # we add an epsilon value to prevent division by zero
    epsilon = 1e-5
    rescaled_x = (x - current_min)/np.max([current_max - current_min , epsilon])
    return rescaled_x

def compute_feature_ranking_double_ended(input, map, fraction):
    
    positives = map[map>0].shape[0]
    total = map.shape[0]*map.shape[1]
    pos_fraction = positives/total
    
#     print("Positives:", positives)
#     print("Total:", total)
#     print("Positive Fraction:", pos_fraction)
    
    if fraction<=pos_fraction:
        
        # Remove from positive end ...
        topk_values = torch.topk(map.view(-1), int(map.view(-1).shape[0]*fraction))

        feature_ranking_positive = torch.zeros_like(map.view(-1))

        feature_ranking_positive[topk_values[1]] = map.view(-1)[topk_values[1]]
        feature_ranking_positive = feature_ranking_positive.reshape(map.shape)
        
        feature_ranking = feature_ranking_positive
        
    elif fraction>pos_fraction:
        
        topk_values = torch.topk(map.view(-1), int(map.view(-1).shape[0]*pos_fraction))

        feature_ranking_positive = torch.zeros_like(map.view(-1))

        feature_ranking_positive[topk_values[1]] = map.view(-1)[topk_values[1]]
        feature_ranking_positive = feature_ranking_positive.reshape(map.shape)
        
        
        # Remove from negative end ...

        topk_values = torch.topk(map.view(-1), int(map.view(-1).shape[0]*(fraction-pos_fraction)), largest=False)

        feature_ranking_negative = torch.zeros_like(map.view(-1))

        feature_ranking_negative[topk_values[1]] = map.view(-1)[topk_values[1]]
        feature_ranking_negative = feature_ranking_negative.reshape(map.shape)
        
        feature_ranking = feature_ranking_positive+feature_ranking_negative

        
    substitute_information = torch.zeros_like(input)
#     substitute_information = torch.fill_(substitute_information, torch.mean(input.float()))
    substitute_information = torch.fill_(substitute_information, 0.0)

    
    
    # Keep only the zero marked entries because they are less important 
    new_data = torch.where(feature_ranking==0, input, substitute_information)
    
    return new_data

def weighted_feature_ranking_double_ended(input, map, fraction, estimator):
    
    positives = map[map>0].shape[0]
    total = map.shape[0]*map.shape[1]
    pos_fraction = positives/total
    
    if fraction<=0.10:
        
        topk_values = torch.topk(map.view(-1), int(map.view(-1).shape[0]*fraction))

        feature_ranking = torch.zeros_like(map.view(-1))

        feature_ranking[topk_values[1]] = map.view(-1)[topk_values[1]]

        feature_ranking = feature_ranking.reshape(map.shape)
        
    
    
    elif fraction<=0.3:
        
        # Remove from positive end ...
             
        topk_values = torch.topk(map.view(-1), int(map.view(-1).shape[0]*fraction*0.90))

        feature_ranking_positive = torch.zeros_like(map.view(-1))

        feature_ranking_positive[topk_values[1]] = map.view(-1)[topk_values[1]]
        feature_ranking_positive = feature_ranking_positive.reshape(map.shape)

        # Remove from negative end ...

        topk_values = torch.topk(map.view(-1), int(map.view(-1).shape[0]*(fraction*0.10)), largest=False)

        feature_ranking_negative = torch.zeros_like(map.view(-1))

        feature_ranking_negative[topk_values[1]] = map.view(-1)[topk_values[1]]
        feature_ranking_negative = feature_ranking_negative.reshape(map.shape)

        feature_ranking = feature_ranking_positive+feature_ranking_negative
        
        
    elif fraction<=0.5:
        
        
        # Remove from positive end ...
             
        topk_values = torch.topk(map.view(-1), int(map.view(-1).shape[0]*fraction*0.90))

        feature_ranking_positive = torch.zeros_like(map.view(-1))

        feature_ranking_positive[topk_values[1]] = map.view(-1)[topk_values[1]]
        feature_ranking_positive = feature_ranking_positive.reshape(map.shape)

        # Remove from negative end ...

        topk_values = torch.topk(map.view(-1), int(map.view(-1).shape[0]*(fraction*0.10)), largest=False)

        feature_ranking_negative = torch.zeros_like(map.view(-1))

        feature_ranking_negative[topk_values[1]] = map.view(-1)[topk_values[1]]
        feature_ranking_negative = feature_ranking_negative.reshape(map.shape)

        feature_ranking = feature_ranking_positive+feature_ranking_negative
        
        
    else:
        
        map = torch.abs(map)
    
        topk_values = torch.topk(map.view(-1), int(map.view(-1).shape[0]*fraction))

        feature_ranking = torch.zeros_like(map.view(-1))

        feature_ranking[topk_values[1]] = map.view(-1)[topk_values[1]]

        feature_ranking = feature_ranking.reshape(map.shape)
            
            
    substitute_information = torch.zeros_like(input)
#     substitute_information = torch.fill_(substitute_information, torch.mean(input.float()))
    substitute_information = torch.fill_(substitute_information, 0.0)
   
    
    # Keep only the zero marked entries because they are less important 
    new_data = torch.where(feature_ranking==0, input, substitute_information)
    
    return new_data


def compute_RAR_feature_ranking(input, map, fraction):
    
    '''
    Retain and Retrain
    '''
   
    topk_values = torch.topk(map.view(-1), int(map.view(-1).shape[0]*fraction))
    
    feature_ranking = torch.zeros_like(map.view(-1))
    
    feature_ranking[topk_values[1]] = map.view(-1)[topk_values[1]]
    
    feature_ranking = feature_ranking.reshape(map.shape)
    
    
    substitute_information = torch.zeros_like(input)
#     substitute_information = torch.fill_(substitute_information, torch.mean(input.float()))
    substitute_information = torch.fill_(substitute_information, 0.0)


    # Keep only the non zero marked entries because they are important as opposed to ROAR
    new_data = torch.where(feature_ranking ==0, substitute_information, input)
    
    return new_data, feature_ranking

def random_RAR_feature_ranking(input, map, fraction):
    
    # fraction says how much we wanna keep 
    
    D = torch.empty(map.shape[0], map.shape[1]).uniform_(0, 1)
    
    substitute_information = torch.zeros_like(input)
#     substitute_information = torch.fill_(substitute_information, torch.mean(input.float()))
    substitute_information = torch.fill_(substitute_information, 0.0)
    
    Yes = torch.ones(map.shape[0], map.shape[1])  #binary mask to keep
    No = torch.zeros(map.shape[0], map.shape[1])  #mask to remove
    
    
    feature_ranking = torch.where(D<=fraction, Yes, No)
    
    new_data = torch.where(feature_ranking==1, input, substitute_information)
    
    return new_data

def random_perm_RAR_feature_ranking(input, map, fraction):
    
    """Random feature selections using random permutation"""
    
    
    P = torch.randperm(map.view(-1).shape[0])
    feature_ranking = torch.zeros_like(map.view(-1))
    
    item_to_keep = int(map.view(-1).shape[0]*fraction)
#     print(item_to_keep)
    
    feature_ranking[P < item_to_keep] = 1
    
    feature_ranking = feature_ranking.reshape(map.shape)
    
    substitute_information = torch.zeros_like(input)
    substitute_information = torch.fill_(substitute_information, 0.0)
    
    new_data = torch.where(feature_ranking==1, input, substitute_information)
    
    return new_data


def compute_feature_ranking(input, map, fraction, estimator):
    
    if estimator == "Grad":
        fraction_threshold = 0.7
    else:
        fraction_threshold = 0.5

#     positives = map[map>0].shape[0]
#     total = map.shape[0]*map.shape[1]
#     pos_fraction = positives/total
    
#     fraction_threshold = pos_fraction
    
    # For grad and sg-grad, use 0.7, for ig and sg-ig, use 0.5
    
    if fraction>fraction_threshold:
        map = torch.abs(map)
    
#     map =  torch.abs(map)
    topk_values = torch.topk(map.view(-1), int(map.view(-1).shape[0]*fraction))
    
    feature_ranking = torch.zeros_like(map.view(-1))
    
    feature_ranking[topk_values[1]] = map.view(-1)[topk_values[1]]
    
    feature_ranking = feature_ranking.reshape(map.shape)
    
#     non_zero_count = torch.nonzero(feature_ranking).size(0)

#     mean = torch.mean(input)
#     std = torch.std(input)    

#     substitute_information = torch.normal(0, 1, size=input.shape)
#     substitute_information = substitute_information*std+mean
    
    substitute_information = torch.zeros_like(input)
#     substitute_information = torch.fill_(substitute_information, torch.mean(input.float()))
    substitute_information = torch.fill_(substitute_information, 0.0)


    # Keep only the zero marked entries because they are less important 
    new_data = torch.where(feature_ranking ==0, input, substitute_information)
    
    return new_data



def random_feature_ranking(input, map, fraction):
    
    # fraction says how much we wanna eliminate 
    
    D = torch.empty(map.shape[0], map.shape[1]).uniform_(0, 1)
    
#     mean = torch.mean(input)
#     std = torch.std(input)    

#     substitute_information = torch.normal(0, 1, size=input.shape)
#     substitute_information = substitute_information*std+mean
    
    substitute_information = torch.zeros_like(input)
#     substitute_information = torch.fill_(substitute_information, torch.mean(input.float()))
    substitute_information = torch.fill_(substitute_information, 0.0)
    
    Yes = torch.ones(map.shape[0], map.shape[1])  #binary mask to keep
    No = torch.zeros(map.shape[0], map.shape[1])  #mask to remove
    
    
    feature_ranking = torch.where(D>fraction, Yes, No)
    
    new_data = torch.where(feature_ranking==1, input, substitute_information)
    
    return new_data
    
    
    
