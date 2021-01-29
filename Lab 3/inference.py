import numpy as np
import graphics
import rover
#%%

def forward_backward(all_possible_hidden_states, all_possible_observed_states, prior_distribution, transition_model, observation_model, observations):
    """
    Inputs
    ------
    all_possible_hidden_states: a list of possible hidden states
    all_possible_observed_states: a list of possible observed states
    prior_distribution: a distribution over states

    transition_model: a function that takes a hidden state and returns a
        Distribution for the next state
    observation_model: a function that takes a hidden state and returns a
        Distribution for the observation from that hidden state
    observations: a list of observations, one per hidden state
        (a missing observation is encoded as None)

    Output
    ------
    A list of marginal distributions at each time step; each distribution
    should be encoded as a Distribution (see the Distribution class in
    rover.py), and the i-th Distribution should correspond to time
    step i
    """

    num_time_steps = len(observations)
    forward_messages = [None] * num_time_steps
    forward_messages[0] = prior_distribution
    backward_messages = [None] * num_time_steps
    marginals = [None] * num_time_steps 
    
    # Compute the forward messages---------------------------------------------
    a_0 = rover.Distribution()
    for state in prior_distribution:
        # a_0 = p(z_0) * p(observation|z_0)
        a_0[state] = prior_distribution[state] * observation_model(state)[observations[0]]
    a_0.renormalize()
    forward_messages[0] = a_0
    
    # Recursively get forward messages
    for i in range( 1, num_time_steps ): #avoid time 0
        a_previous = forward_messages[i-1] #known
        a_current = rover.Distribution() #to be found. 
        
        for state in all_possible_hidden_states:
            
            # sum of a(z_i-1) * p(zi | zi-1)
            sum = 0
            for prev_state in a_previous:
                # a(z_i-1) * p(zi | zi-1)
                sum += a_previous[prev_state] * transition_model(prev_state)[state]
            
            # Handle missing observations
            if observations[i] == None:
                # 1 * p[ (xi, yi) | zi ]
                p = sum
            else:
                # sum * p[ (xi, yi) | zi ]
                p = observation_model(state)[ observations[i] ] * sum
            
            # Insert into forward messages. Only update if something changes => faster               
            if p != 0:
                # sum * p[ (xi, yi) | zi ]
                a_current[state] = p
        
        # Renormalize function modified to avoid division by 0
        a_current.renormalize()
        forward_messages[i] = a_current
 
                   
    # Compute the backward messages--------------------------------------------
    b_N_1 = rover.Distribution() #b(z_n-1)
    for state in all_possible_hidden_states:
        b_N_1[state] = 1
    
    b_N_1.renormalize()
    backward_messages[num_time_steps - 1] = b_N_1
    
    # Recursively get backward messages, starting with N-1 to 0
    for i in range( num_time_steps - 1, 0, -1 ):
        b_next = backward_messages[i] # already calculated
        b_current = rover.Distribution() # we need to find this
        
        for curr_state in all_possible_hidden_states:
            sum = 0
            for next_state in b_next:
                
                if observations[i] == None:
                    # Sum of b(zi) * 1 * p( zi | zi-1 )
                    sum += b_next[next_state] * transition_model(curr_state)[next_state]
                
                else:
                    # Sum of b(zi) * p( observation | zi ) * p( zi | zi-1 )
                    sum += b_next[next_state] * observation_model(next_state)[ observations[i] ] * transition_model(curr_state)[next_state]
            
            # Only update if values changed => faster
            if sum != 0:
                b_current[curr_state] = sum
        
        b_current.renormalize()
        backward_messages[i - 1] = b_current
            
    
    # Compute the marginals----------------------------------------------------
    for i in range(num_time_steps):
        gamma_zi = rover.Distribution()
        
        for state in all_possible_hidden_states:
            # g(zi) = a(zi) * b(zi)
            gamma_zi[state] = forward_messages[i][state] * backward_messages[i][state]
        
        gamma_zi.renormalize()
        marginals[i] = gamma_zi
            
    return marginals

#%%
def safelog(x):
    if x == 0:
        return -np.inf
    else:
        return np.log(x)
#%%
def Viterbi(all_possible_hidden_states, all_possible_observed_states,prior_distribution,transition_model,observation_model,observations):
    """
    Inputs
    ------
    See the list inputs for the function forward_backward() above.

    Output
    ------
    A list of esitmated hidden states, each state is encoded as a tuple
    (<x>, <y>, <action>)
    """
    num_time_steps = len(observations)
    w = [None] * num_time_steps
    estimated_hidden_states = [None] * num_time_steps
    
    # Get wo = log[ p(observations|z0)*p(z0) ]
    wo = rover.Distribution()
    
    for state in all_possible_hidden_states:
        # Handle missing data
        if observations[0] == None:
            wo[state] = safelog( 1 ) + safelog(prior_distribution[state])
        else:
            wo[state] = safelog( observation_model(state)[observations[0]] ) + safelog(prior_distribution[state])
            
    w[0] = wo
    estimated_hidden_states[0] = w[0].get_mode() #NOTE: slight mod to get_mode to work w -inf
    
    
    # Get w for each time step
    for i in range(1, num_time_steps): #avoid i = 0
        w_prev = w[i-1] #known
        w_current = rover.Distribution() #to find
        
        # wi(zi) = log[ p(observation|state) ] + max{ log[p(zi|zi-1)] + wi-1(zi-1) }
        for state in all_possible_hidden_states:
            
            # Handle missing data
            if observations[i] == None:
                term1 = safelog( 1 ) #log[ p(observation|state) ] = 1
            else:
                term1 = safelog( observation_model(state)[observations[i]] ) #log[ p(observation|state) ]
            
            # prev_state to maximize { log[p(zi|zi-1)] + wi-1(zi-1) }
            max = -np.inf
            for prev_state in w_prev:
                term2 = safelog( transition_model(prev_state)[state] ) #log[p(zi|zi-1)]
                term3 = w_prev[prev_state] #wi-1(zi-1)
                
                if term2 + term3 > max:
                    max = term2 + term3
            
            w_current[state] = term1 + max
        
        w[i] = w_current
        
        # Find the most likely state and insert into list
        estimated_hidden_states[i] = w[i].get_mode()
    
    return estimated_hidden_states
#%%
if __name__ == '__main__':
   
    enable_graphics = False
    
    missing_observations = True
    if missing_observations:
        filename = 'test_missing.txt'
    else:
        filename = 'test.txt'
            
    # load data    
    hidden_states, observations = rover.load_data(filename)
    num_time_steps = len(hidden_states)

    all_possible_hidden_states   = rover.get_all_hidden_states()
    all_possible_observed_states = rover.get_all_observed_states()
    prior_distribution           = rover.initial_distribution()
    
    print('Running forward-backward...')
    marginals = forward_backward(all_possible_hidden_states, all_possible_observed_states,prior_distribution,rover.transition_model,rover.observation_model, observations)
    print('\n')
   
    timestep = num_time_steps - 1
    print("Most likely parts of marginal at time %d:" % (timestep))
    print(sorted(marginals[timestep].items(), key=lambda x: x[1], reverse=True)[:10])
    print('\n')
    
    timestep = 30
    print("Most likely parts of marginal at time %d:" % (timestep))
    print(sorted(marginals[timestep].items(), key=lambda x: x[1], reverse=True)[:10])
    print('\n')

    print('Running Viterbi...')
    estimated_states = Viterbi(all_possible_hidden_states,all_possible_observed_states,prior_distribution,rover.transition_model,rover.observation_model,observations)
    print('\n')
    
    print("Last 10 hidden states in the MAP estimate:")
    for time_step in range(num_time_steps - 10, num_time_steps):
        print(estimated_states[time_step])

    error_forwardbackward = 0
    error_viterbi = 0
    for i in range(num_time_steps):
        if estimated_states[i] != hidden_states[i]:
            error_viterbi += 1
            
        if marginals[i].get_mode() != hidden_states[i]:
            error_forwardbackward += 1
        
    print("Error using forward/backward:", error_forwardbackward/num_time_steps)
    print("Error using Viterbi:", error_viterbi/num_time_steps)
    
    for i in range(num_time_steps-1):
        print("Time:", i, "  Position:", marginals[i].get_mode()[:2], "  prev:", marginals[i].get_mode()[2])

    # if you haven't complete the algorithms, to use the visualization tool
    # let estimated_states = [None]*num_time_steps, marginals = [None]*num_time_steps
    # estimated_states = [None]*num_time_steps
    # marginals = [None]*num_time_steps
    if enable_graphics:
        app = graphics.playback_positions(hidden_states,observations,estimated_states,marginals)
        app.mainloop()
        
