# Guided Performance Tuning

```
   ________  __________  __________      ________  ___   _______   ________
  / ____/ / / /  _/ __ \/ ____/ __ \    /_  __/ / / / | / /  _/ | / / ____/
 / / __/ / / // // / / / __/ / / / /_____/ / / / / /  |/ // //  |/ / / __  
/ /_/ / /_/ // // /_/ / /___/ /_/ /_____/ / / /_/ / /|  // // /|  / /_/ /  
\____/\____/___/_____/_____/_____/     /_/  \____/_/ |_/___/_/ |_/\____/   
                                                                           
```

## General

**Guided Tuning** (GT), is designed to provide a streamlined and efficient approach to optimize performance of HPC and ML apps. By leveraging a tuned set of high-level performance counters and guided workflows, this project aims to simplify the iterative nature of performance profiling. GT currently supports the following architectures:

- **AMD Instinct**: MI200, MI300

For more information on available features, installation steps, and workload profiling and analysis, please refer to the online [documentation](https://didactic-adventure-j7p5zgz.pages.github.io/).

## Testing

To quickly verify that the GT installation is working correctly, you can run the following command:

```bash
pytest tests/test_project.py -vv
```

See [install instructions](https://didactic-adventure-j7p5zgz.pages.github.io/installation.html#install) for more information on prerequisites and installation steps.
