"use strict";
var canvas = document.querySelector("canvas");
var context = canvas.getContext("2d");
var size = 15;

var width = canvas.width;
var height = canvas.height;
var pixels = width < height ? width : height;
var unit = pixels / (size + 2);
var rect = canvas.getBoundingClientRect();

var user_input = -1;
var waiting    = false;
var fst = null, snd = null;

addEventListener("load", function(event) {
    var req = new XMLHttpRequest();
    var moves = 0;

    req.addEventListener("load", function() {
        var obj = JSON.parse(req.responseText);
        size = obj["size"]; 
        unit = pixels / (size + 2);
        moves = obj["moves"];
        if (moves % 2 == 0) {
            fst  = obj["fst"];
            snd  = obj["snd"];
        } else {
            fst  = obj["snd"];
            snd  = obj["fst"];
        }
        drawboard(context, unit, size);
        var newreq = new XMLHttpRequest();
        newreq.addEventListener("load", request_received);

        newreq.open("GET", "next", true);
        newreq.send(null);
    })
    req.open("GET", "info", true);
    req.send(null);
})

addEventListener("click", function(event) {
    var y = Math.round((event.clientX - rect.left) / unit) - 1;
    var x = Math.round((event.clientY - rect.top) / unit) - 1;
    if (x >= 0 && x < size && y >= 0 && y < size) {
        if (user_input == -1 && waiting == true) {
            user_input = x * size + y;
            var req = new XMLHttpRequest();
            req.addEventListener("load", function() {
                waiting = false;
                user_input = -1;
                req = new XMLHttpRequest();
                req.addEventListener("load", request_received);
                req.open("GET", "next", true);
                req.send(null);
            });
            req.open("GET", "move?move="+user_input, true);
            req.send(null);
        }
    }
});

var color = true;

function request_received() {
    var req = this;
    var obj = JSON.parse(req.responseText);
    var newreq = false;
    var timeout = 100;
    switch (obj["result"]) {
        case "clear":   drawboard(context, unit, size);
                        user_input = -1;
                        waiting = false;
                        newreq = true;
                        color = true;
                        break;
        case "none":    waiting = true;
                        break;
        case "end":     var connected = obj["highlights"].sort();
                        var last = -1;
                        for (var i = 0; i < connected.length; i ++) {
                            if (last != connected[i]) {
                                var x = connected[i] % size;
                                var y = Math.floor(connected[i] / size);
                                highlights(context, x, y, unit, color);
                            }
                            last = connected[i];
                        }
                        timeout = 5000;
                        newreq = true;
                        break;
        case "move":
                        var m = obj["move"];
                        drawstone(context, m[1], m[0], unit, color);
                        color = !color;
                        newreq = true;
                        break;
    }
    if (newreq) {
        setTimeout(function() {
            var req = new XMLHttpRequest();
            req.addEventListener("load", request_received);
            req.open("GET", "next", true);
            req.send(null);
        }, timeout);
    }
}

function highlights(context, x, y, unit, color) {
    var _x = (x + 1) * unit, _y = (y + 1) * unit;
    context.beginPath();
    context.moveTo(_x, _y);
    context.arc(_x, _y, Math.floor(unit / 5), 0, 2 * Math.PI);
    context.fillStyle =color ? "black" : "white";
    context.fill();
}

function drawstone(context, x, y, unit, color) {
    context.fillStyle = color ? "black" : "white";
    var _x = (x + 1) * unit, _y = (y + 1) * unit;
    context.beginPath();
    context.moveTo(_x, _y);
    context.arc(_x, _y, Math.floor(unit / 5 * 2), 0, 2 * Math.PI);
    context.strokeStyle = "black";
    context.stroke();
    context.fill();
}

function drawboard(context, unit, size) {
    context.fillStyle = "#ffffcc";
    context.fillRect(0, 0, (size + 2) * unit, (size + 2) * unit);
    context.beginPath();
    for (var x = unit; x <= unit * (size + 1); x += unit) {
        context.moveTo(x, unit);
        context.lineTo(x, unit * size);
    }
    for (var x = unit; x <= unit * (size + 1); x += unit) {
        context.moveTo(unit, x);
        context.lineTo(unit * size, x);
    }
    function dot(x, y, unit) {
        var _x = unit * (1 + x);
        var _y = unit * (1 + y);
        context.moveTo(_x, _y);
        context.arc(_x, _y, Math.floor(unit / 10), 0, 2 * Math.PI);
    }
    var dots = [3, Math.floor(size / 2), size - 4];
    for (var x = 0; x < dots.length; x ++)
        for (var y = 0; y < dots.length; y ++)
            dot(dots[x], dots[y], unit);
    context.strokeStyle = "black";
    context.stroke();
    var fst_moves = 0;
    if (fst != null) {
        for (var i = 0; i < size * size; i ++) {
            if (fst[i]) {
                var y = Math.floor(i / size)
                var x = i % size
                fst_moves ++;
                drawstone(context, x, y, unit, true);
            }
        }
        fst = null
    }
    var snd_moves = 0;
    if (snd != null) {
        for (var i = 0; i < size * size; i ++) {
            if (snd[i]) {
                var y = Math.floor(i / size)
                var x = i % size
                snd_moves ++;
                drawstone(context, x, y, unit, false);
            }
        }
        snd = null
    }
    if (snd_moves == fst_moves)
        color = true;
    else
        color = false;
}
