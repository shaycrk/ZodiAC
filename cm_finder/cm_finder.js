
var watchID;

function watchLocation() {
    document.getElementById("start-button").setAttribute("style", "display: none;");
    document.getElementById("log-button").setAttribute("style", "display: block;");
    document.getElementById("demo").setAttribute("style", "display: block;");
	if (navigator.geolocation) {
	  const options = {
		enableHighAccuracy: true,
		maximumAge: 0,
		timeout: 27000,
	  };
	  watchID = navigator.geolocation.watchPosition(update_dom, showError, options);
      document.getElementById('chime-audio').play(); 
      document.getElementById('chime-audio').pause(); 
      document.getElementById('chime-audio').currentTime = 0;
	  return watchID;
	} else {
      var x = document.getElementById("demo");
	  x.innerHTML = "Geolocation is not supported by this browser.";
	  return null;
	}
}
  
  function showPosition(position) {
    var x = document.getElementById("demo");
	x.innerHTML = "Latitude: " + position.coords.latitude +
	"<br>Longitude: " + position.coords.longitude +
	"<br>Accuracy (m): " + position.coords.accuracy +
	"<br>Heading: " + position.coords.heading;
  }
  
  function showError(error) {
    var x = document.getElementById("demo");
	switch(error.code) {
	  case error.PERMISSION_DENIED:
		x.innerHTML = "User denied the request for Geolocation."
		break;
	  case error.POSITION_UNAVAILABLE:
		x.innerHTML = "Location information is unavailable."
		break;
	  case error.TIMEOUT:
		x.innerHTML = "The request to get user location timed out."
		break;
	  case error.UNKNOWN_ERROR:
		x.innerHTML = "An unknown error occurred."
		break;
	}
  }


// could use google maps spherical api, but not clear if use of the functions
// actually constitute API calls, which could be a ton during the course of 
// a rallye

function deg2rad(deg) {
  return deg * (Math.PI/180)
}
 
// Converts from radians to degrees.
function rad2deg(radians) {
  return radians * 180 / Math.PI;
}

function getDistanceFromLatLon(lat1, lon1, lat2, lon2) {
  var R = 6378137; // Radius of the earth in m
  var dLat = deg2rad(lat2-lat1);  // deg2rad below
  var dLon = deg2rad(lon2-lon1); 
  var a = 
    Math.sin(dLat/2) * Math.sin(dLat/2) +
    Math.cos(deg2rad(lat1)) * Math.cos(deg2rad(lat2)) * 
    Math.sin(dLon/2) * Math.sin(dLon/2)
    ; 
  var c = 2 * Math.atan2(Math.sqrt(a), Math.sqrt(1-a)); 
  var d = R * c; // Distance in m
  return d;
}

function bearing(startLat, startLng, destLat, destLng){
  var startLatRad = deg2rad(startLat);
  var startLngRad = deg2rad(startLng);
  var destLatRad = deg2rad(destLat);
  var destLngRad = deg2rad(destLng);

  var br_y = Math.sin(destLngRad - startLngRad) * Math.cos(destLatRad);
  var br_x = Math.cos(startLatRad) * Math.sin(destLatRad) -
        Math.sin(startLatRad) * Math.cos(destLatRad) * Math.cos(destLngRad - startLngRad);
  var brng = Math.atan2(br_y, br_x);
  brng = rad2deg(brng);
  return (brng + 360) % 360;
}


// tightening down the CM_DIST parameter might help a lot with noisiness (40 seemed quite good with
// logged data from walking in Redwood City), but need to get some test data from driving to have
// a better sense of the right values here since the speed (and thus distance between samples)
// will be higher...
const CM_DIST = 50; // meters
const CM_HEADING = 20; // degrees
const CM_LAT_DIST = 25; // meters

var tracking_log = [];
const MAX_LOG = 10000; // maximum number of tracked locations to log

// less than 150 m, heading within 20 degrees

var locs = [
	[38.524573, -121.969126],
	[38.524925, -121.968148],
	[38.525175, -121.967467],
  [38.525465, -121.966678],
	[38.525662, -121.966093],
  [38.525868, -121.965519],
  [38.525945, -121.965303], 
  [38.526174, -121.964696],
  [38.526384, -121.964117],
  [38.526602, -121.963891],
  [38.526904, -121.963183],
  [38.527215, -121.962336],
  [38.527483, -121.961456]
];

locs = locs.reverse();

const CMs = [
  [38.525653, -121.967800, "A\n27", 350],
  [38.526014, -121.965279, "E\n66", 260],
  [38.524990, -121.968197, "H\n20", 260],
  [38.525519, -121.96601, "PP\n6", 80],
  [38.524763, -121.967306, "U\n1", 170],
  [38.524025, -121.968251, "Y\n10", 260],
  [38.523143, -121.968272, "X\n25", 260],
  [38.523236, -121.966394, "B\n9", 170],
  [38.522480, -121.967714, "HHH\n19", 170],
  [38.524033, -121.970895, "G\n14", 260],
  [38.522315, -121.970109, "K\n17", 80],
  [38.520947, -121.969970, "SS\n21", 350],
  [38.522736, -121.971073, "R\n33", 170],
  [38.523126, -121.971269, "I\n22", 170],
  [38.523490, -121.971481, "N\n23", 350],
  [38.524104, -121.971849, "II\n3", 350],
  [38.523439, -121.972470, "O\n12", 260],
  [38.521254, -121.971947, "T\n11", 350],
  [38.520161, -121.969709, "SSS\n31", 170],
  [38.519739, -121.972135, "F\n4", 80],
  [38.521471, -121.973899, "Z\n666", 170],
  [38.523507, -121.971790, "EE\n32", 80],
  [38.521599, -121.975002, "YY\n7", 260],
  [38.520154, -121.975059, "V\n8", 170],
  [38.520404, -121.975786, "W\n29", 80],
  [38.520097, -121.976635, "J\n26", 260],
  [38.519240, -121.975990, "M\n13", 80],
  [38.518633, -121.975247, "D\n28", 260],
  [38.518039, -121.974593, "AA\n2", 170],
  [38.518173, -121.976235, "III\n30", 260],
  [38.518940, -121.978580, "BNG\n31", 350],
  [38.518250, -121.978784, "Q\n16", 80],
  [38.518518, -121.979364, "S\n5", 310],
  [38.518122, -121.980834, "P\n18", 260],
  [38.519656, -121.981234, "HH\n15", 350],
  [38.483035, -122.000249, "L\n34", 10],

  // some locations in Redwood City
  [37.482233, -122.228655, "ZZ\n123", 40],
  [37.481986, -122.227566, "RWC\n45", 130],
  [37.482331, -122.228049, "YY\n89", 310],
  [37.482207, -122.226439, "MPL\n20", 5]
];

var prev_loc = {coords: {latitude: locs[0][0], longitude: locs[0][1]}};
var prev_hdng = 250;

//const loc1 = {lat: -34, lng: 151};
//const loc2 = {lat: -34, lng: 151.00001}

//const x = document.getElementById("demo");

const USE_API_HEADING = false;

// new
const HDNG_BOXCAR_WINDOW = 5;
var hdng_boxcar_num = [];
var hdng_boxcar_denom = [];
var calculated_headings = [];

function update_heading(new_loc) {
  var hdng = null;
  var dist_to_last = getDistanceFromLatLon(
  	new_loc.coords.latitude, new_loc.coords.longitude, 
    prev_loc.coords.latitude, prev_loc.coords.longitude
  );
  // prefer the heading from geolocation if it's available
  // (seems more stable generally but to have a little trouble with U-turns?)
  // also assumes every device will return heading -- might need some check if
  // this is persistently null to fall back to the other logic?
  if ((USE_API_HEADING) && ('heading' in new_loc.coords)) {
    //hdng = new_loc.coords.heading ?? prev_hdng;

    // new (also uncomment above)
    if (new_loc.coords.heading != null) {
      hdng_boxcar_num.push(Math.sin(deg2rad(new_loc.coords.heading)));
      hdng_boxcar_denom.push(Math.cos(deg2rad(new_loc.coords.heading)));
      if (hdng_boxcar_num.length > HDNG_BOXCAR_WINDOW) {
          hdng_boxcar_num = hdng_boxcar_num.slice(hdng_boxcar_num.length - HDNG_BOXCAR_WINDOW, hdng_boxcar_num.length);
          hdng_boxcar_denom = hdng_boxcar_denom.slice(hdng_boxcar_denom.length - HDNG_BOXCAR_WINDOW, hdng_boxcar_denom.length);
      }
      num = hdng_boxcar_num.reduce((a,b) => a+b, 0);
      denom = hdng_boxcar_denom.reduce((a,b) => a+b, 0);
      hdng = (rad2deg(Math.atan2(num, denom)) + 360) % 360;
    } else {
      hdng = prev_hdng;
    }

    prev_loc = new_loc; // update the location regardless of whether heading is available?
    prev_hdng = hdng;
  }
  else if (dist_to_last > 6) {
  	var new_hdng = bearing(
        prev_loc.coords.latitude, prev_loc.coords.longitude, 
        new_loc.coords.latitude, new_loc.coords.longitude
    );

    hdng_boxcar_num.push(Math.sin(deg2rad(new_hdng)));
    hdng_boxcar_denom.push(Math.cos(deg2rad(new_hdng)));
    if (hdng_boxcar_num.length > HDNG_BOXCAR_WINDOW) {
        hdng_boxcar_num = hdng_boxcar_num.slice(hdng_boxcar_num.length - HDNG_BOXCAR_WINDOW, hdng_boxcar_num.length);
        hdng_boxcar_denom = hdng_boxcar_denom.slice(hdng_boxcar_denom.length - HDNG_BOXCAR_WINDOW, hdng_boxcar_denom.length);
    }
    num = hdng_boxcar_num.reduce((a,b) => a+b, 0);
    denom = hdng_boxcar_denom.reduce((a,b) => a+b, 0);
    hdng = (rad2deg(Math.atan2(num, denom)) + 360) % 360;

    prev_loc = new_loc;
    prev_hdng = hdng;
  }
  else {
  	hdng = prev_hdng;
  }
  calculated_headings.push(hdng); //new
  return [hdng, dist_to_last];
}

// lots of nested ifs but avoids doing extra calculations every location update
function find_cms(ltln, hdng) {
	var nearest_cm = {dist: 100000000};
	for (var i = 0; i < CMs.length; i++) {

		// need to be traveling in the direction CM is facing
        var rel_hdng = Math.min(
            (hdng - CMs[i][3] + 360) % 360,
            (CMs[i][3] - hdng + 360) % 360
        );
	    if (rel_hdng <= CM_HEADING) {
            var cm_loc = {coords: {latitude: CMs[i][0], longitude: CMs[i][1]}};
            var dist = getDistanceFromLatLon(
                ltln.coords.latitude, ltln.coords.longitude, 
                cm_loc.coords.latitude, cm_loc.coords.longitude
            );
            if ((dist <= CM_DIST) && (dist < nearest_cm['dist'])) {
                var cm_hdng = bearing(
                    ltln.coords.latitude, ltln.coords.longitude, 
                    cm_loc.coords.latitude, cm_loc.coords.longitude
                );
                // also need to be approaching the CM, not past it
                // as you get parallel to the CM, this will approach +/- 90 degrees, and as you
                // move past it the angle will grow larger (using 85 to provide some buffer)
                //
                // but also need to ensure the CM is on this street not on a parallel one, so
                // check that the component of the distance vector perpendicular to the direction
                // of travel isn't more than the width of a wide road (e.g., a large relative)
                // angle is fine if we're right by the CM but not if it's far away on a separate
                // street!
                var rel_cm_hdng = Math.min(
                    (hdng - cm_hdng + 360) % 360,
                    (cm_hdng - hdng + 360) % 360
                );
                var lateral_dist = dist * Math.sin(deg2rad(rel_cm_hdng));
                if ((rel_cm_hdng <= 85) && (lateral_dist <= CM_LAT_DIST)) {
                    nearest_cm = {
                        dist: dist,
                        cm: CMs[i],
                        cm_hdng: cm_hdng
                    };
                }
			}
        }
  }
  return nearest_cm;
}

//var play_count = 0;

function update_dom(location) {
    var x = document.getElementById("demo");
    var [hdng, dist_to_last] = update_heading(location);

    var nearest_cm = find_cms(location, hdng);

    if (tracking_log.length < MAX_LOG) { 
        tracking_log.push(location);
        // JSON.stringify of the tracked locations giving empty elements on the iphone,
        // so try just dumping directly to the DOM...
        add_to_log_div(location);
    }
    /*
    x.innerHTML = "<br /><br />Distance to last: " + dist_to_last + 
    "<br />Heading: " + hdng;
    */

    /*
    // Testing async sound trigger...
    if (('accuracy' in location.coords) && (play_count < 30) && (play_count % 10 == 0)) {
        document.getElementById('chime-audio').play();
    }
    document.getElementById("play-cnt").innerHTML = "Play Count: " + play_count;
    play_count++;
    */

    if  ('cm' in nearest_cm) {
        var cm_dist_ft = Math.round(nearest_cm['dist'] * 3.28);
        x.innerHTML = "<CENTER><H1>Coursemarker Found!<br /><br />" +
        nearest_cm['cm'][2].replace('\n', '<br />') + "</H1>" +
        "<H2><br /><br /> About " + cm_dist_ft + " feet ahead</H2></CENTER>";
        /*
        x.innerHTML += "<br />Distance to CM: " + nearest_cm['dist'] + 
        "<br />Heading to CM: " + nearest_cm['cm_hdng'] +
        "<br />CM Found: " + nearest_cm['cm'][2];
        */

        var new_last_cm = "Last CM Seen: " + nearest_cm['cm'][2];
        if (document.getElementById("demo").style.display == 'none') {
            document.getElementById("searching").setAttribute("style", "display: none;");
            document.getElementById("demo").setAttribute("style", "display: block;");
            
            document.getElementById('chime-audio').play();
        } else if (document.getElementById("lastcm").innerHTML != new_last_cm) {
            // play chime if we've seen a new CM even if already on the CM page
            document.getElementById('chime-audio').play();
        }

        document.getElementById("lastcm").innerHTML = new_last_cm;

    } else {
        document.getElementById("search-pos").innerHTML = "<br />Current Location: (" +
            location.coords.latitude.toFixed(6) + ", " + location.coords.longitude.toFixed(6) +")" +
            "<br />Heading: " + hdng.toFixed(1) + " degrees";
        if ('accuracy' in location.coords) {
            document.getElementById("search-pos").innerHTML += "<br />Accuracy (m): " + location.coords.accuracy +
            "<br />Heading (geolocation): " + location.coords.heading +
            "<br />Updated: " + (new Date(location.timestamp).toLocaleString());
        }
        document.getElementById("demo").setAttribute("style", "display: none;");
        document.getElementById("searching").setAttribute("style", "display: block;");
    }
}


function test_rallye() {
  document.getElementById("start-button").setAttribute("style", "display: none;");
  document.getElementById("log-button").setAttribute("style", "display: block;");
  document.getElementById("searching").setAttribute("style", "display: block;");

  document.getElementById('chime-audio').play(); 
  document.getElementById('chime-audio').pause(); 
  document.getElementById('chime-audio').currentTime = 0;

  for (let i = 0; i < locs.length; i++) {
  	setTimeout(function() {
      var ltln = {coords: {latitude: locs[i][0], longitude: locs[i][1]}};
      //var dist0 = google.maps.geometry.spherical.computeDistanceBetween(ltln, cm_loc);
      //var hdng0 = google.maps.geometry.spherical.computeHeading(ltln, cm_loc);
      update_dom(ltln);
    }, 0+3000*i);
  }
}


function test_log_rallye() {
  document.getElementById("start-button").setAttribute("style", "display: none;");
  document.getElementById("log-button").setAttribute("style", "display: block;");
  document.getElementById("searching").setAttribute("style", "display: block;");

  document.getElementById('chime-audio').play(); 
  document.getElementById('chime-audio').pause(); 
  document.getElementById('chime-audio').currentTime = 0;

  prev_loc = log_locs[0];
  prev_hdng = 135;

  for (let i = 0; i < log_locs.length; i++) {
  	setTimeout(function() {
      var ltln = log_locs[i];
      var old_view = document.getElementById("searching").style.display == 'block' ? 'search' : 'cm';
      update_dom(ltln);
      var new_view = document.getElementById("searching").style.display == 'block' ? 'search' : 'cm';
      if (old_view != new_view) {
        console.log("" + i + ": " + old_view + " => " + new_view);
      }
    }, 0+250*i);
  }
}


function add_to_log_div(location) {
    document.getElementById("tracking-log").textContent += "{" + 
        "latitude: " + location.coords.latitude +
        ", longitude: " + location.coords.longitude;
    if ('accuracy' in location.coords) {
        document.getElementById("tracking-log").textContent += ", accuracy: " + location.coords.accuracy +
            ", heading: " + location.coords.heading +
            ", timestamp: " + location.timestamp +
            "},"
    } else {
        document.getElementById("tracking-log").textContent += "},";
    }

}

function show_log() {
    if (watchID != null) {
        navigator.geolocation.clearWatch(watchID);
    }
    //const newText = document.createTextNode(JSON.stringify(tracking_log));
    //document.getElementById("searching").innerHTML = "";
    //document.getElementById("searching").appendChild(newText);

    document.getElementById("start-button").setAttribute("style", "display: none;");
    document.getElementById("log-button").setAttribute("style", "display: none;");
    document.getElementById("demo").setAttribute("style", "display: none;");
    document.getElementById("searching").setAttribute("style", "display: none;");

    document.getElementById("tracking-log").setAttribute("style", "display: block;");

    // try coping to clipboard as well for convenience...
    navigator.clipboard.writeText(document.getElementById("tracking-log").textContent);
}




