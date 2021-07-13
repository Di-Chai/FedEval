## Server-side HTTP Interface

### Status

#### Req

http://{ServerHostAddr}/status

#### Res

```json
{
	'finished': bool,	// whether the training process end or not
	'rounds': number,	// ith round that we are running at (+2 bias?)
	'log_dir': str,		// the directory path of all the log & results files received from containers
}
```

### DashBoard

#### Req

http://{ServerHostAddr}/dashboard

#### Res

A web page about the running session. Specifically, it's of:

* Status: Running or Finished
* Rounds: the current round number and its maximum limitation
* Clients: Train / Online / Require
* Time Usage: time that have been spend at current session
* Best Test Accuracy & Loss: the best accuracy & loss in tests up to now
* Server Send & Receive: the traffic in and out of the main server (?)
* Metrics for Validations & Tests: two graph demonstrating 

A refresh is required if you want to have the most recent results on the page.

### Download

#### Req

http://{ServerHostAddr}/download/{filename}

#### Res

Download the file as an attachment.