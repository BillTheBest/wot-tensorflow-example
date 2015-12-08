from wot import Wot
import numpy as np

def echo(message,meta,headers):
	print("Label: %s" % meta)
	print(np.loads(message))
	

w = Wot("amqp://test:test@localhost:5672/wot")
w.start( [ 
	( w.new_channel, []),
	( w.stream_resource, [ "mnist", echo ]) 
])
