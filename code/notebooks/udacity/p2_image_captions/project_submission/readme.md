[original project github](https://github.com/udacity/CVND---Image-Captioning-Project)
## Reference Paper
[Show and Tell: A Neural Image Caption Generator](https://arxiv.org/pdf/1411.4555.pdf)
## Keeping Your Session Active
The workspace_utils.py module (available here) includes an iterator wrapper called keep_awake and a context manager called active_session that can be used to maintain an active session during long-running processes. The two functions are equivalent, so use whichever fits better in your code.

```
from workspace_utils import keep_awake

for i in keep_awake(range(5)):  #anything that happens inside this loop will keep the workspace active
    # do iteration with lots of work here
```
```
from workspace_utils import active_session

with active_session():
    # do long-running work here
```
![Tux, the Linux mascot](../images/cnn_rnn_arch.png)
