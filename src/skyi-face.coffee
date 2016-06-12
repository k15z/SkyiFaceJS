SVM = require("svm").SVM;

class SkyiFace
    constructor: (width = 96, height = 96, bin_width = 12, bin_height = 12) ->
        @data = {}
        @model = {}
        @width = width
        @height = height
        @bin_width = bin_width
        @bin_height = bin_height
        @num_bins_x = parseInt(width/bin_width)
        @num_bins_y = parseInt(height/bin_height)

    ### Compute all of the one-vs-all SVMs ###
    compute: () ->
        for name, _ of @data
            labels = []
            vectors = []
            for key, value of @data
                for vector in value
                    if key == name 
                        labels.push(1)
                    else
                        labels.push(-1)
                    vectors.push(vector)
            @model[name] = new SVM()
            @model[name].train(vectors, labels, {numpasses: 100})
        @ready = true
        return

    ### Vectorize and append to data set ###
    memorize: (name, canvas) ->
        @ready = false
        if not @data[name]
            @data[name] = []
        @data[name].push(@_vectorize(canvas))
        return

    ### Recompute if necessary, then run canvas against each SVM ###
    recognize: (canvas) ->
        result = {}
        vector = @_vectorize(canvas)
        if not @ready
            @compute()
        for key, svm of @model
            result[key] = svm.marginOne(vector)
        total = (Math.exp(value) for key, value of result).reduce((t, s) -> t + s)
        for key, value of result
            result[key] = {
                margin: value
                probability: Math.exp(value) / total
            }
        return result

    ### Compute local binary pattern histogram vector ###
    _vectorize: (canvas) ->
        original = canvas
        canvas = document.createElement('canvas')
        canvas.width = @width
        canvas.height = @height
        context = canvas.getContext('2d')
        context.drawImage(original, 0, 0, original.width, original.height, 0, 0, canvas.width, canvas.height)
        data = context.getImageData(0, 0, canvas.width, canvas.height).data

        pixel = (0 for [0...(@width*@height)])
        for x in [0...@width]
            for y in [0...@height]
                pixel[y*@width+x] = (data[(y*@width+x)*4+0] + data[(y*@width+x)*4+1] + data[(y*@width+x)*4+2])/(255*3)

        pattern = (0 for [0...(@width*@height)])
        for x in [1...@width-1]
            for y in [1...@height-1]
                value = 0
                if pixel[y*@width+x] > pixel[(y-1)*@width+(x+0)] then value |= (0x1 << 0)
                if pixel[y*@width+x] > pixel[(y-1)*@width+(x+1)] then value |= (0x1 << 1)
                if pixel[y*@width+x] > pixel[(y+0)*@width+(x+1)] then value |= (0x1 << 2)
                if pixel[y*@width+x] > pixel[(y+1)*@width+(x+1)] then value |= (0x1 << 3)
                if pixel[y*@width+x] > pixel[(y+1)*@width+(x-0)] then value |= (0x1 << 4)
                if pixel[y*@width+x] > pixel[(y+1)*@width+(x-1)] then value |= (0x1 << 5)
                if pixel[y*@width+x] > pixel[(y-0)*@width+(x-1)] then value |= (0x1 << 6)
                if pixel[y*@width+x] > pixel[(y-1)*@width+(x-1)] then value |= (0x1 << 7)
                pattern[y*@width+x] = value

        vector = []
        for x in [0...@num_bins_x]
            for y in [0...@num_bins_y]
                histogram = (0 for [0...256])
                for dx in [0...@bin_width]
                    for dy in [0...@bin_height]
                        histogram[pattern[(y*@bin_height+dy)*@width+(x*@bin_width+dx)]]++;
                vector = vector.concat(value / (@bin_width * @bin_height) for value in histogram)
        return vector

window.SkyiFace = SkyiFace
