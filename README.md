# skyi-face.js
Offline javascript face recognition using local binary patterns and support vector machines. The 
running time should scale linearly with number of faces, increasing by around 10 ms for each 
additional person to consider in Google Chrome v51.0 on my 2015 i5 Surface Book.

Note: When you call `memorizeFace`, the canvas is vectorized and stored, but the models are NOT 
updated until you call `recognizeFace`. This means the first time you call recognizeFace, there 
will be a significant delay before it returns a result - future calls will not have this delay 
and will run thousands of times faster. If you want to train/update the models manually, you can 
manually call the `skyi.compute()` function.

## Usage: skyi-face.js
```
var skyi = new SkyiFace();

skyi.memorizeFace("person 1", canvas)
skyi.memorizeFace("person 1", canvas)
// ... more samples = better
skyi.memorizeFace("person 1", canvas)

skyi.memorizeFace("person 2", canvas)
skyi.memorizeFace("person 2", canvas)
// ... more samples = better
skyi.memorizeFace("person 2", canvas)

skyi.recognizeFace(canvas)
```
