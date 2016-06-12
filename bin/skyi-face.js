(function e(t,n,r){function s(o,u){if(!n[o]){if(!t[o]){var a=typeof require=="function"&&require;if(!u&&a)return a(o,!0);if(i)return i(o,!0);var f=new Error("Cannot find module '"+o+"'");throw f.code="MODULE_NOT_FOUND",f}var l=n[o]={exports:{}};t[o][0].call(l.exports,function(e){var n=t[o][1][e];return s(n?n:e)},l,l.exports,e,t,n,r)}return n[o].exports}var i=typeof require=="function"&&require;for(var o=0;o<r.length;o++)s(r[o]);return s})({1:[function(require,module,exports){
// MIT License
// Andrej Karpathy

var svmjs = (function(exports){

  /*
    This is a binary SVM and is trained using the SMO algorithm.
    Reference: "The Simplified SMO Algorithm" (http://math.unt.edu/~hsp0009/smo.pdf)
    
    Simple usage example:
    svm = svmjs.SVM();
    svm.train(data, labels);
    testlabels = svm.predict(testdata);
  */
  var SVM = function(options) {
  }

  SVM.prototype = {
    
    // data is NxD array of floats. labels are 1 or -1.
    train: function(data, labels, options) {
      
      // we need these in helper functions
      this.data = data;
      this.labels = labels;

      // parameters
      options = options || {};
      var C = options.C || 1.0; // C value. Decrease for more regularization
      var tol = options.tol || 1e-4; // numerical tolerance. Don't touch unless you're pro
      var alphatol = options.alphatol || 1e-7; // non-support vectors for space and time efficiency are truncated. To guarantee correct result set this to 0 to do no truncating. If you want to increase efficiency, experiment with setting this little higher, up to maybe 1e-4 or so.
      var maxiter = options.maxiter || 10000; // max number of iterations
      var numpasses = options.numpasses || 10; // how many passes over data with no change before we halt? Increase for more precision.
      
      // instantiate kernel according to options. kernel can be given as string or as a custom function
      var kernel = linearKernel;
      this.kernelType = "linear";
      if("kernel" in options) {
        if(typeof options.kernel === "string") {
          // kernel was specified as a string. Handle these special cases appropriately
          if(options.kernel === "linear") { 
            this.kernelType = "linear"; 
            kernel = linearKernel; 
          }
          if(options.kernel === "rbf") { 
            var rbfSigma = options.rbfsigma || 0.5;
            this.rbfSigma = rbfSigma; // back this up
            this.kernelType = "rbf";
            kernel = makeRbfKernel(rbfSigma);
          }
        } else {
          // assume kernel was specified as a function. Let's just use it
          this.kernelType = "custom";
          kernel = options.kernel;
        }
      }

      // initializations
      this.kernel = kernel;
      this.N = data.length; var N = this.N;
      this.D = data[0].length; var D = this.D;
      this.alpha = zeros(N);
      this.b = 0.0;
      this.usew_ = false; // internal efficiency flag

      // run SMO algorithm
      var iter = 0;
      var passes = 0;
      while(passes < numpasses && iter < maxiter) {
        
        var alphaChanged = 0;
        for(var i=0;i<N;i++) {
        
          var Ei= this.marginOne(data[i]) - labels[i];
          if( (labels[i]*Ei < -tol && this.alpha[i] < C)
           || (labels[i]*Ei > tol && this.alpha[i] > 0) ){
            
            // alpha_i needs updating! Pick a j to update it with
            var j = i;
            while(j === i) j= randi(0, this.N);
            var Ej= this.marginOne(data[j]) - labels[j];
            
            // calculate L and H bounds for j to ensure we're in [0 C]x[0 C] box
            ai= this.alpha[i];
            aj= this.alpha[j];
            var L = 0; var H = C;
            if(labels[i] === labels[j]) {
              L = Math.max(0, ai+aj-C);
              H = Math.min(C, ai+aj);
            } else {
              L = Math.max(0, aj-ai);
              H = Math.min(C, C+aj-ai);
            }
            
            if(Math.abs(L - H) < 1e-4) continue;
            
            var eta = 2*kernel(data[i],data[j]) - kernel(data[i],data[i]) - kernel(data[j],data[j]);
            if(eta >= 0) continue;
            
            // compute new alpha_j and clip it inside [0 C]x[0 C] box
            // then compute alpha_i based on it.
            var newaj = aj - labels[j]*(Ei-Ej) / eta;
            if(newaj>H) newaj = H;
            if(newaj<L) newaj = L;
            if(Math.abs(aj - newaj) < 1e-4) continue; 
            this.alpha[j] = newaj;
            var newai = ai + labels[i]*labels[j]*(aj - newaj);
            this.alpha[i] = newai;
            
            // update the bias term
            var b1 = this.b - Ei - labels[i]*(newai-ai)*kernel(data[i],data[i])
                     - labels[j]*(newaj-aj)*kernel(data[i],data[j]);
            var b2 = this.b - Ej - labels[i]*(newai-ai)*kernel(data[i],data[j])
                     - labels[j]*(newaj-aj)*kernel(data[j],data[j]);
            this.b = 0.5*(b1+b2);
            if(newai > 0 && newai < C) this.b= b1;
            if(newaj > 0 && newaj < C) this.b= b2;
            
            alphaChanged++;
            
          } // end alpha_i needed updating
        } // end for i=1..N
        
        iter++;
        //console.log("iter number %d, alphaChanged = %d", iter, alphaChanged);
        if(alphaChanged == 0) passes++;
        else passes= 0;
        
      } // end outer loop
      
      // if the user was using a linear kernel, lets also compute and store the
      // weights. This will speed up evaluations during testing time
      if(this.kernelType === "linear") {

        // compute weights and store them
        this.w = new Array(this.D);
        for(var j=0;j<this.D;j++) {
          var s= 0.0;
          for(var i=0;i<this.N;i++) {
            s+= this.alpha[i] * labels[i] * data[i][j];
          }
          this.w[j] = s;
          this.usew_ = true;
        }
      } else {

        // okay, we need to retain all the support vectors in the training data,
        // we can't just get away with computing the weights and throwing it out

        // But! We only need to store the support vectors for evaluation of testing
        // instances. So filter here based on this.alpha[i]. The training data
        // for which this.alpha[i] = 0 is irrelevant for future. 
        var newdata = [];
        var newlabels = [];
        var newalpha = [];
        for(var i=0;i<this.N;i++) {
          //console.log("alpha=%f", this.alpha[i]);
          if(this.alpha[i] > alphatol) {
            newdata.push(this.data[i]);
            newlabels.push(this.labels[i]);
            newalpha.push(this.alpha[i]);
          }
        }

        // store data and labels
        this.data = newdata;
        this.labels = newlabels;
        this.alpha = newalpha;
        this.N = this.data.length;
        //console.log("filtered training data from %d to %d support vectors.", data.length, this.data.length);
      }

      var trainstats = {};
      trainstats.iters= iter;
      return trainstats;
    }, 
    
    // inst is an array of length D. Returns margin of given example
    // this is the core prediction function. All others are for convenience mostly
    // and end up calling this one somehow.
    marginOne: function(inst) {

      var f = this.b;
      // if the linear kernel was used and w was computed and stored,
      // (i.e. the svm has fully finished training)
      // the internal class variable usew_ will be set to true.
      if(this.usew_) {

        // we can speed this up a lot by using the computed weights
        // we computed these during train(). This is significantly faster
        // than the version below
        for(var j=0;j<this.D;j++) {
          f += inst[j] * this.w[j];
        }

      } else {

        for(var i=0;i<this.N;i++) {
          f += this.alpha[i] * this.labels[i] * this.kernel(inst, this.data[i]);
        }
      }

      return f;
    },
    
    predictOne: function(inst) { 
      return this.marginOne(inst) > 0 ? 1 : -1; 
    },
    
    // data is an NxD array. Returns array of margins.
    margins: function(data) {
      
      // go over support vectors and accumulate the prediction. 
      var N = data.length;
      var margins = new Array(N);
      for(var i=0;i<N;i++) {
        margins[i] = this.marginOne(data[i]);
      }
      return margins;
      
    },
    
    // data is NxD array. Returns array of 1 or -1, predictions
    predict: function(data) {
      var margs = this.margins(data);
      for(var i=0;i<margs.length;i++) {
        margs[i] = margs[i] > 0 ? 1 : -1;
      }
      return margs;
    },
    
    // THIS FUNCTION IS NOW DEPRECATED. WORKS FINE BUT NO NEED TO USE ANYMORE. 
    // LEAVING IT HERE JUST FOR BACKWARDS COMPATIBILITY FOR A WHILE.
    // if we trained a linear svm, it is possible to calculate just the weights and the offset
    // prediction is then yhat = sign(X * w + b)
    getWeights: function() {
      
      // DEPRECATED
      var w= new Array(this.D);
      for(var j=0;j<this.D;j++) {
        var s= 0.0;
        for(var i=0;i<this.N;i++) {
          s+= this.alpha[i] * this.labels[i] * this.data[i][j];
        }
        w[j]= s;
      }
      return {w: w, b: this.b};
    },

    toJSON: function() {
      
      if(this.kernelType === "custom") {
        console.log("Can't save this SVM because it's using custom, unsupported kernel...");
        return {};
      }

      json = {}
      json.N = this.N;
      json.D = this.D;
      json.b = this.b;

      json.kernelType = this.kernelType;
      if(this.kernelType === "linear") { 
        // just back up the weights
        json.w = this.w; 
      }
      if(this.kernelType === "rbf") { 
        // we need to store the support vectors and the sigma
        json.rbfSigma = this.rbfSigma; 
        json.data = this.data;
        json.labels = this.labels;
        json.alpha = this.alpha;
      }

      return json;
    },
    
    fromJSON: function(json) {
      
      this.N = json.N;
      this.D = json.D;
      this.b = json.b;

      this.kernelType = json.kernelType;
      if(this.kernelType === "linear") { 

        // load the weights! 
        this.w = json.w; 
        this.usew_ = true; 
        this.kernel = linearKernel; // this shouldn't be necessary
      }
      else if(this.kernelType == "rbf") {

        // initialize the kernel
        this.rbfSigma = json.rbfSigma; 
        this.kernel = makeRbfKernel(this.rbfSigma);

        // load the support vectors
        this.data = json.data;
        this.labels = json.labels;
        this.alpha = json.alpha;
      } else {
        console.log("ERROR! unrecognized kernel type." + this.kernelType);
      }
    }
  }
  
  // Kernels
  function makeRbfKernel(sigma) {
    return function(v1, v2) {
      var s=0;
      for(var q=0;q<v1.length;q++) { s += (v1[q] - v2[q])*(v1[q] - v2[q]); } 
      return Math.exp(-s/(2.0*sigma*sigma));
    }
  }
  
  function linearKernel(v1, v2) {
    var s=0; 
    for(var q=0;q<v1.length;q++) { s += v1[q] * v2[q]; } 
    return s;
  }

  // Misc utility functions
  // generate random floating point number between a and b
  function randf(a, b) {
    return Math.random()*(b-a)+a;
  }

  // generate random integer between a and b (b excluded)
  function randi(a, b) {
     return Math.floor(Math.random()*(b-a)+a);
  }

  // create vector of zeros of length n
  function zeros(n) {
    var arr= new Array(n);
    for(var i=0;i<n;i++) { arr[i]= 0; }
    return arr;
  }

  // export public members
  exports = exports || {};
  exports.SVM = SVM;
  exports.makeRbfKernel = makeRbfKernel;
  exports.linearKernel = linearKernel;
  return exports;

})(typeof module != 'undefined' && module.exports);  // add exports to module.exports if in node.js

},{}],2:[function(require,module,exports){
// Generated by CoffeeScript 1.10.0
(function() {
  var SVM, SkyiFace;

  SVM = require("svm").SVM;

  SkyiFace = (function() {
    function SkyiFace(width, height, bin_width, bin_height) {
      if (width == null) {
        width = 96;
      }
      if (height == null) {
        height = 96;
      }
      if (bin_width == null) {
        bin_width = 12;
      }
      if (bin_height == null) {
        bin_height = 12;
      }
      this.data = {};
      this.model = {};
      this.width = width;
      this.height = height;
      this.bin_width = bin_width;
      this.bin_height = bin_height;
      this.num_bins_x = parseInt(width / bin_width);
      this.num_bins_y = parseInt(height / bin_height);
    }


    /* Compute all of the one-vs-all SVMs */

    SkyiFace.prototype.compute = function() {
      var _, i, key, labels, len, name, ref, ref1, value, vector, vectors;
      ref = this.data;
      for (name in ref) {
        _ = ref[name];
        labels = [];
        vectors = [];
        ref1 = this.data;
        for (key in ref1) {
          value = ref1[key];
          for (i = 0, len = value.length; i < len; i++) {
            vector = value[i];
            if (key === name) {
              labels.push(1);
            } else {
              labels.push(-1);
            }
            vectors.push(vector);
          }
        }
        this.model[name] = new SVM();
        this.model[name].train(vectors, labels, {
          numpasses: 100
        });
      }
      this.ready = true;
    };


    /* Vectorize and append to data set */

    SkyiFace.prototype.memorize = function(name, canvas) {
      this.ready = false;
      if (!this.data[name]) {
        this.data[name] = [];
      }
      this.data[name].push(this._vectorize(canvas));
    };


    /* Recompute if necessary, then run canvas against each SVM */

    SkyiFace.prototype.recognize = function(canvas) {
      var key, ref, result, svm, total, value, vector;
      result = {};
      vector = this._vectorize(canvas);
      if (!this.ready) {
        this.compute();
      }
      ref = this.model;
      for (key in ref) {
        svm = ref[key];
        result[key] = svm.marginOne(vector);
      }
      total = ((function() {
        var results;
        results = [];
        for (key in result) {
          value = result[key];
          results.push(Math.exp(value));
        }
        return results;
      })()).reduce(function(t, s) {
        return t + s;
      });
      for (key in result) {
        value = result[key];
        result[key] = {
          margin: value,
          probability: Math.exp(value) / total
        };
      }
      return result;
    };


    /* Compute local binary pattern histogram vector */

    SkyiFace.prototype._vectorize = function(canvas) {
      var context, data, dx, dy, histogram, i, j, k, l, m, n, o, original, p, pattern, pixel, ref, ref1, ref2, ref3, ref4, ref5, ref6, ref7, value, vector, x, y;
      original = canvas;
      canvas = document.createElement('canvas');
      canvas.width = this.width;
      canvas.height = this.height;
      context = canvas.getContext('2d');
      context.drawImage(original, 0, 0, original.width, original.height, 0, 0, canvas.width, canvas.height);
      data = context.getImageData(0, 0, canvas.width, canvas.height).data;
      pixel = (function() {
        var i, ref, results;
        results = [];
        for (i = 0, ref = this.width * this.height; 0 <= ref ? i < ref : i > ref; 0 <= ref ? i++ : i--) {
          results.push(0);
        }
        return results;
      }).call(this);
      for (x = i = 0, ref = this.width; 0 <= ref ? i < ref : i > ref; x = 0 <= ref ? ++i : --i) {
        for (y = j = 0, ref1 = this.height; 0 <= ref1 ? j < ref1 : j > ref1; y = 0 <= ref1 ? ++j : --j) {
          pixel[y * this.width + x] = (data[(y * this.width + x) * 4 + 0] + data[(y * this.width + x) * 4 + 1] + data[(y * this.width + x) * 4 + 2]) / (255 * 3);
        }
      }
      pattern = (function() {
        var k, ref2, results;
        results = [];
        for (k = 0, ref2 = this.width * this.height; 0 <= ref2 ? k < ref2 : k > ref2; 0 <= ref2 ? k++ : k--) {
          results.push(0);
        }
        return results;
      }).call(this);
      for (x = k = 1, ref2 = this.width - 1; 1 <= ref2 ? k < ref2 : k > ref2; x = 1 <= ref2 ? ++k : --k) {
        for (y = l = 1, ref3 = this.height - 1; 1 <= ref3 ? l < ref3 : l > ref3; y = 1 <= ref3 ? ++l : --l) {
          value = 0;
          if (pixel[y * this.width + x] > pixel[(y - 1) * this.width + (x + 0)]) {
            value |= 0x1 << 0;
          }
          if (pixel[y * this.width + x] > pixel[(y - 1) * this.width + (x + 1)]) {
            value |= 0x1 << 1;
          }
          if (pixel[y * this.width + x] > pixel[(y + 0) * this.width + (x + 1)]) {
            value |= 0x1 << 2;
          }
          if (pixel[y * this.width + x] > pixel[(y + 1) * this.width + (x + 1)]) {
            value |= 0x1 << 3;
          }
          if (pixel[y * this.width + x] > pixel[(y + 1) * this.width + (x - 0)]) {
            value |= 0x1 << 4;
          }
          if (pixel[y * this.width + x] > pixel[(y + 1) * this.width + (x - 1)]) {
            value |= 0x1 << 5;
          }
          if (pixel[y * this.width + x] > pixel[(y - 0) * this.width + (x - 1)]) {
            value |= 0x1 << 6;
          }
          if (pixel[y * this.width + x] > pixel[(y - 1) * this.width + (x - 1)]) {
            value |= 0x1 << 7;
          }
          pattern[y * this.width + x] = value;
        }
      }
      vector = [];
      for (x = m = 0, ref4 = this.num_bins_x; 0 <= ref4 ? m < ref4 : m > ref4; x = 0 <= ref4 ? ++m : --m) {
        for (y = n = 0, ref5 = this.num_bins_y; 0 <= ref5 ? n < ref5 : n > ref5; y = 0 <= ref5 ? ++n : --n) {
          histogram = (function() {
            var o, results;
            results = [];
            for (o = 0; o < 256; o++) {
              results.push(0);
            }
            return results;
          })();
          for (dx = o = 0, ref6 = this.bin_width; 0 <= ref6 ? o < ref6 : o > ref6; dx = 0 <= ref6 ? ++o : --o) {
            for (dy = p = 0, ref7 = this.bin_height; 0 <= ref7 ? p < ref7 : p > ref7; dy = 0 <= ref7 ? ++p : --p) {
              histogram[pattern[(y * this.bin_height + dy) * this.width + (x * this.bin_width + dx)]]++;
            }
          }
          vector = vector.concat((function() {
            var len, q, results;
            results = [];
            for (q = 0, len = histogram.length; q < len; q++) {
              value = histogram[q];
              results.push(value / (this.bin_width * this.bin_height));
            }
            return results;
          }).call(this));
        }
      }
      return vector;
    };

    return SkyiFace;

  })();

  window.SkyiFace = SkyiFace;

}).call(this);

},{"svm":1}]},{},[2]);
