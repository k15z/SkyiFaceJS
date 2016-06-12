fs = require('fs')
rimraf = require('rimraf')
exec = require('child_process').exec

task 'sbuild', 'standard build', (options) ->
    exec 'coffee --compile --output temp/ src/', (err, stdout, stderr) ->
        throw err if err
        console.log stdout + stderr

        exec 'browserify temp/skyi-face.js > bin/skyi-face.js', (err, stdout, stderr) ->
            throw err if err
            console.log stdout + stderr

            rimraf 'temp', () ->
                return
