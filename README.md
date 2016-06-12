# skyi-face.js
Offline javascript face recognition using local binary patterns and support vector machines.

## Usage
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

## Setup
```
git clone <repository>
npm install
cake sbuild
```
