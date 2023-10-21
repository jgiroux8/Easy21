import numpy as np
from Easy21 import Easy21
from Sarsa import Sarsa
from MonteCarlo import MC_Control
from utils import plot_value_function,plot_learning_curve,plot_mse_lambda
import json
import os
import argparse


def main(config):
    episodes = config['episodes']
    lambda_values = config['lambda_values']
    gamma = config['gamma']
    
    if os.path.exists('Figures'):
        print('Figures folder exists, overwriting files in the folder.')
    else:
        os.mkdir('Figures')
    
    print('Creating Easy21 Environment')
    env = Easy21()
    
    if bool(config['run_MC']):
        print('Running MC Control with {0} episodes.'.format(episodes))
        mc_control = MC_Control(env)
        Q_mc = mc_control.run_MC(episodes)
        plot_value_function(Q_mc,episodes,'MC_Control')
    else:
        print('MC not enabled. Set boolean flag in config file.')
        Q_mc = None
        
        
    if bool(config['run_sarsa']):
        print('Running Sarsa for lambda values between {0}-{1}'.format(lambda_values[0],lambda_values[-1]))
        
        mse_list = []
        final_mse = []
        for lamb in lambda_values:
            sarsa = Sarsa(env,lamb,gamma,Q_star=Q_mc)
            
            Q,mse = sarsa.run_sarsa(episodes)
            mse_list.append(mse)
            final_mse.append(mse[-1][1])
            plot_value_function(Q,episodes,'Sarsa({0})'.format(lamb))
            
        if Q_mc is not None:
            print('Plotting learning curve for all lambdas.')
            plot_learning_curve(mse_list,lambda_values)
            print('Plotting MSE as a function of lambda.')
            plot_mse_lambda(final_mse,lambda_values)
        
    else:
        print('Sarsa not enabled. Set boolean flag in config file.')
    
    
    return -1
    
    
    
    
if __name__ =='__main__':
    parser = argparse.ArgumentParser(description='Easy21 RL')
    parser.add_argument('-c','--config',default='config.json',type=str,
                        help='Path to the config file (default: config.json)')
                        
    args = parser.parse_args()
    
    config = json.load(open(args.config))
    
    main(config)

    
