#It's the Heuristic Rule-based Human Local Guide


class HeuriRules:
    def  complex_rules(obs,action):
            for k in (0,len(obs)-1):
                #If it is in the local area
                if(obs[k]['image'][3][-2][0] == 9):
                      has_hole = 0
                      for l in (0,len(obs[k]['image'])-1):
                               hole = obs[k]['image'][l][-2][0]
                               #If there is a hole, change the direction towards the "1"(hole)
                               if(hole == 1):
                                   has_hole = 1
                                   if l < 3:
                                       action[k] = 0
                                   if l > 3:
                                       action[k] = 1
                                   break
                      #If there is no hole, towards the opposite to the "0"(none) or "2"(wall) 
                      if(has_hole == 0):
                           if(obs[k]['image'][0][-2][0] == 0 or obs[k]['image'][0][-2][0] == 2):
                               action[k] = 1
                           else:
                               action[k] = 0
            return action
   
   # The simple_rule is to change the direction when the agent want to go into the obs '9'(lava)
   def simple_rule(obs,action)
        for k in (0,len(obs)-1):
            if(obs[k]['image'][3][-2][0] == 9 and action[k] == 2):
              action[k] = 1
        return action


