# GA Strategy Optimizer

Note: This is an older project that needed to be updated and uploaded because its on my CV.

Just a simple project to learn about genetic algorithms. The idea is to use evolution-inspired optimization to find good trading strategy parameters. I'm not a quant, just thought it would be cool to see if a GA could find better moving average crossover settings than random guessing.

The strategy is basic - buy when short MA crosses above long MA, sell when it crosses below. The GA optimizes the window sizes by testing different combinations and keeping the ones with better returns across generations. You can see the ROI improving over generations in the plot it generates.
