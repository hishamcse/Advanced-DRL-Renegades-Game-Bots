## Diambra Agent

### Directory Structure

   * **Final Street Fighter III Training Agent Script**: final-agent.py
   * All other scripts are self explanatory. (Filename describes about each script)

### Commands
  
   * Run with default config: 
   ```diambra run -r "absolute roms path" python basic-env.py```
   * Run using 2 parallel environment: ```diambra run -r "absolute roms path" -s=2 python parallel-ppo-agent.py```
   * Run with custom config: 
      ```diambra run -r "{absolute roms path}" python final-agent.py --cfgFile "{absolute yaml config path}"```
   * Submit Agent: Follow this [link](https://github.com/alexpalms/deep-rl-class/blob/main/units/en/unitbonus3/agent-submission.mdx)


### Resources

   * Tutorial: https://github.com/alexpalms/deep-rl-class/blob/main/units/en/unitbonus3

   * Full Diambra Tutorial: https://diambra.gitlab.io/website/downloadenv/#tutorials

   * Documentation: https://docs.diambra.ai/

   * Example Agents: https://github.com/diambra/agents/tree/main

   * Youtube Tutorial: https://www.youtube.com/watch?v=PJhRRv9rwOg&ab_channel=JousefMuradLITE

   * Sample Agent with Submission: https://github.com/mscrnt/Diambra_Agent/blob/funsize