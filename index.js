/**
 * Created by frederickmacgregor on 20/04/2017.
 */
"use strict";

// taken from http://karpathy.github.io/neuralnets/

var Unit = function(value, gradient){
    this.value = value;
    this.grad = gradient;
};



var multiplyGate = function(){};


multiplyGate.prototype.forward = function(u0, u1) {
    this.u0 = u0;
    this.u1 = u1;
    this.utop = new Unit(u0.value * u1.value, 0);
    return this.utop;
};

multiplyGate.prototype.backward = function(){
    this.u0.grad += this.u1.value * this.utop.grad;
    this.u1.grad += this.u0.value * this.utop.grad;
};


var addGate = function () {

};

addGate.prototype.forward = function(u0, u1){
    this.u0 = u0;
    this.u1 = u1;
    this.utop = new Unit(u0.value + u1.value, 0);
    return this.utop;
};

addGate.prototype.backward = function(){
    this.u0.grad += 1 * this.utop.grad;
    this.u1.grad += 1 * this.utop.grad;
};

var maxGate = function() {

};

maxGate.prototype.forward = function (u0, u1) {
    this.u0 = u0;
    this.u1 = u1;
    // console.log("max this.u0", this.u0);
    // console.log("max this.u1", this.u1);
    this.utop = new Unit(this.u0.value > this.u1.value ? this.u0.value : this.u1.value, 0);
    return this.utop;
};

maxGate.prototype.backward = function() {
  this.u0.grad =  this.u0.value > this.u1.value ? this.utop.grad : 0;
  this.u1.grad =  this.u1.value > this.u0.value ? this.utop.grad : 0;
};


var sigmoidGate = function(){
    this.sig = function(x){
        return 1/ (1 + Math.exp(-x));
    }
};

sigmoidGate.prototype.forward = function(u0){
    this.u0 = u0;
    this.utop = new Unit(this.sig(this.u0.value), 0);
    return this.utop;
};

sigmoidGate.prototype.backward = function(){
    var s = this.sig(this.u0.value);
    this.u0.grad += (s * (1-s)) * this.utop.grad;
};

// create input units
var a = new Unit(1.0, 0.0);
var b = new Unit(2.0, 0.0);
var c = new Unit(-3.0, 0.0);
var x = new Unit(-1.0, 0.0);
var y = new Unit(3.0, 0.0);

// create the gates
var mulg0 = new multiplyGate();
var mulg1 = new multiplyGate();
var addg0 = new addGate();
var addg1 = new addGate();
var sg0 = new sigmoidGate();

// do the forward pass
var forwardNeuron = function() {
    var ax = mulg0.forward(a, x), // a*x = -1
    by = mulg1.forward(b, y), // b*y = 6
    axpby = addg0.forward(ax, by), // a*x + b*y = 5
    axpbypc = addg1.forward(axpby, c), // a*x + b*y + c = 2
    s = sg0.forward(axpbypc); // sig(a*x + b*y + c) = 0.8808
    // console.log('ax', ax);
    // console.log('by', by);
    // console.log('axpby', axpby);
    // console.log('axpbypc', axpbypc);
    // console.log('sig', s.value);
    s.grad = 1;
    sg0.backward();
    addg1.backward();
    addg0.backward();
    mulg1.backward();
    mulg0.backward();
    return s;
};
var s = forwardNeuron();

// console.log('circuit output: ' + s.value); // prints 0.8808


var step_size = 0.01;
a.value += step_size * a.grad; // a.grad is -0.105
b.value += step_size * b.grad; // b.grad is 0.315
c.value += step_size * c.grad; // c.grad is 0.105
x.value += step_size * x.grad; // x.grad is 0.105
y.value += step_size * y.grad; // y.grad is 0.210

// console.log('a', a);
// console.log('b', b);
// console.log('c', c);
// console.log('x', x);
// console.log('y', y);

// s = forwardNeuron();
// console.log('circuit output after one backprop: ' + s.value); // prints 0.8825


// A circuit: it takes 5 Units (x,y,a,b,c) and outputs a single Unit
// It can also compute the gradient w.r.t. its inputs
var Circuit = function() {
    // create some gates
    this.mulg0 = new multiplyGate();
    this.mulg1 = new multiplyGate();
    this.addg0 = new addGate();
    this.addg1 = new addGate();
};
Circuit.prototype = {
    forward: function(x,y,a,b,c) {
        this.ax = this.mulg0.forward(a, x); // a*x
        this.by = this.mulg1.forward(b, y); // b*y
        this.axpby = this.addg0.forward(this.ax, this.by); // a*x + b*y
        this.axpbypc = this.addg1.forward(this.axpby, c); // a*x + b*y + c
        return this.axpbypc;
    },
    backward: function(gradient_top) { // takes pull from above
        this.axpbypc.grad = gradient_top;
        this.addg1.backward(); // sets gradient in axpby and c
        this.addg0.backward(); // sets gradient in ax and by
        this.mulg1.backward(); // sets gradient in b and y
        this.mulg0.backward(); // sets gradient in a and x
    }
};

var Circuit2 = function() {
    // create some gates
    this.mulg0 = new multiplyGate();
    this.mulg1 = new multiplyGate();
    this.addg0 = new addGate();
    this.addg1 = new addGate();
    this.maxg0 = new maxGate();

    this.mulg2 = new multiplyGate();
    this.mulg3 = new multiplyGate();
    this.addg2 = new addGate();
    this.addg3 = new addGate();
    this.maxg1 = new maxGate();

    this.mulg4 = new multiplyGate();
    this.mulg5 = new multiplyGate();
    this.addg4 = new addGate();
    this.addg5 = new addGate();
    this.maxg2 = new maxGate();

    this.mulg6 = new multiplyGate();
    this.mulg7 = new multiplyGate();
    this.mulg8 = new multiplyGate();
    this.addg6 = new addGate();
    this.addg7 = new addGate();
    this.addg8 = new addGate();
};
Circuit2.prototype = {
    forward: function(x,y,params) {
        if (params.length < 13){
            return console.log("error param length less than 13: ", params.length);
        }
        this.params = params;

        var n1, n2, n3;
        // this.params.forEach((param) => {
        //     param.
        // });
        this.ax = this.mulg0.forward(params[0], x); // a*x
        this.by = this.mulg1.forward(params[1], y); // b*y
        this.axpby = this.addg0.forward(this.ax, this.by); // a*x + b*y
        this.axpbypc = this.addg1.forward(this.axpby, params[2]); // a*x + b*y + c
        // console.log("this.axpbypc", this.axpbypc);
        this.maxaxpbypc = n1 = this.maxg0.forward(new Unit(0,0), this.axpbypc);

        this.ax1 = this.mulg2.forward(params[3], x); // a*x
        this.by1 = this.mulg3.forward(params[4], y); // b*y
        this.axpby1 = this.addg2.forward(this.ax1, this.by1); // a*x + b*y
        this.axpbypc1 = this.addg3.forward(this.axpby1, params[5]); // a*x + b*y + c
        this.maxaxpbypc1 = n2 = this.maxg1.forward(new Unit(0,0), this.axpbypc1);

        this.ax2 = this.mulg4.forward(params[6], x); // a*x
        this.by2 = this.mulg5.forward(params[7], y); // b*y
        this.axpby2 = this.addg4.forward(this.ax2, this.by2); // a*x + b*y
        this.axpbypc2 = this.addg5.forward(this.axpby2, params[8]); // a*x + b*y + c
        this.maxaxpbypc2 = n3 = this.maxg2.forward(new Unit(0,0), this.axpbypc2);
        // console.log("N1", n1);
        this.axn1 = this.mulg6.forward(params[9], n1); // a*x
        this.bxn2 = this.mulg7.forward(params[10], n2); // b*y
        this.bxn3 = this.mulg8.forward(params[11], n3); // b*y
        this.axn1pbxn2 = this.addg6.forward(this.axn1, this.bxn2); // a*x + b*y
        this.axxxxpbxn3 = this.addg7.forward(this.axn1pbxn2, this.bxn3); // a*x + b*y
        this.score = this.addg8.forward(this.axxxxpbxn3, params[12]); // a*x + b*y + c

        return this.score;
    },
    backward: function(gradient_top) { // takes pull from above
        this.score.grad = gradient_top;
        this.addg8.backward();
        this.addg7.backward();
        this.addg6.backward();
        this.mulg8.backward();
        this.mulg7.backward();
        this.mulg6.backward();
        this.maxg2.backward();
        this.addg5.backward();
        this.addg4.backward();
        this.mulg5.backward();
        this.mulg4.backward();
        this.maxg1.backward();
        this.addg3.backward();
        this.addg2.backward();
        this.mulg3.backward();
        this.mulg2.backward();
        this.maxg0.backward();
        this.addg1.backward(); // sets gradient in axpby and c
        this.addg0.backward(); // sets gradient in ax and by
        this.mulg1.backward(); // sets gradient in b and y
        this.mulg0.backward(); // sets gradient in a and x
    }
};


var SVM = function() {

    // random initial parameter values
    this.a = new Unit(1.0, 0.0);
    this.b = new Unit(-2.0, 0.0);
    this.c = new Unit(-1.0, 0.0);
    this.params = [];

    for (var i = 0; i<13; i++){
        this.params.push(new Unit(2 + Math.random()*-1, 0));
    }

    console.log("params: ", this.params);

    this.circuit = new Circuit2();
};
SVM.prototype = {
    forward: function(x, y) { // assume x and y are Units
        this.unit_out = this.circuit.forward(x, y, this.params);
        return this.unit_out;
    },
    backward: function(label) { // label is +1 or -1

        // reset pulls on a,b,c
        this.params.forEach(function(param){
            param.grad = 0;
        });

        // compute the pull based on what the circuit output was
        var pull = 0.0;
        if(label === 1 && this.unit_out.value < 1) {
            pull = 1; // the score was too low: pull up
        }
        if(label === -1 && this.unit_out.value > -1) {
            pull = -1; // the score was too high for a positive example, pull down
        }
        this.circuit.backward(pull); // writes gradient into x,y,a,b,c

        // add regularization pull for parameters: towards zero and proportional to value

        this.params.forEach(function(param){
            param.grad += -param.value;
        });
    },
    learnFrom: function(x, y, label) {
        this.forward(x, y); // forward pass (set .value in all Units)
        this.backward(label); // backward pass (set .grad in all Units)
        this.parameterUpdate(); // parameters respond to tug
    },
    parameterUpdate: function() {
        var step_size = 0.01;
        this.params.forEach(function(param){
            param.value += step_size * param.grad;
        });
    }
};

// now let's train

var data = []; var labels = [];
data.push([1.2, 0.7]); labels.push(1);
data.push([-0.3, -0.5]); labels.push(-1);
data.push([3.0, 0.1]); labels.push(1);
data.push([-0.1, -1.0]); labels.push(-1);
data.push([-1.0, 1.1]); labels.push(-1);
data.push([2.1, -3]); labels.push(1);
var svm = new SVM();

// a function that computes the classification accuracy
var evalTrainingAccuracy = function() {
    var num_correct = 0;
    for(var i = 0; i < data.length; i++) {
        var x = new Unit(data[i][0], 0.0);
        var y = new Unit(data[i][1], 0.0);
        var true_label = labels[i];

        // see if the prediction matches the provided label
        var predicted_label = svm.forward(x, y).value > 0 ? 1 : -1;
        if(predicted_label === true_label) {
            num_correct++;
        }
    }
    return num_correct / data.length;
};

// the learning loop
var learn = function(){
    for(var iter = 0; iter < 400; iter++) {
        // pick a random data point
        var i = Math.floor(Math.random() * data.length);
        var x = new Unit(data[i][0], 0.0);
        var y = new Unit(data[i][1], 0.0);
        var label = labels[i];
        svm.learnFrom(x, y, label);

        if(iter % 25 == 0) { // every 10 iterations...
            console.log('training accuracy at iter ' + iter + ': ' + evalTrainingAccuracy());
            if (evalTrainingAccuracy() === 1){
                console.log("Success! values:", svm.params);
                break;
            }
        }
    }
};
learn();
