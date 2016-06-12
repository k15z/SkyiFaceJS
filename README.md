# skyi-face.js
Offline javascript face recognition using local binary patterns and support vector machines. The 
running time should scale linearly with number of faces, increasing by around 10 ms for each 
additional person to consider in Google Chrome v51.0 on my 2015 i5 Surface Book.

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
