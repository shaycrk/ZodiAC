# ZodiAC
A virtual car rallye interface via google maps streetview!

There are three main files of interest here:
- **rallye.html**: A tool for running car rallyes (specified in JSON format) using google streetview
- **rallye_builder.html**: A tool for creating or editing these rallyes
- **dad_route66_rallye.json**: Config file for the Route 66-themed rallye my mom wrote for my dad's 66th birthday

## Future Work / Known Issues
There are a few outstanding features that would be nice to have:
- Pass through other objects when editing a json that contains them (even if these can't be edited/shown on the map, would be good not to lose them)

- Currently you have to hover over coursemarkers to see their labels in a tooltip, but it would be nice to actually have these display on the pie plate itself. This might be do-able with a google.maps.Marker.label, but will take some experimentation (espeically to work with different zoom levels)

- Allow for custom object types (with custom images) in the rallye builder (there's already support for them in the rallye runner itself, but need a builder interface with additional options, espeically to get image sizing right)

- Presently, we don't support mobile (or even tablet) devices. It will take some research to see what making the rallyes mobile-friendly would entail (probably including at least: dealing with the streetview interface that seems to want to be full screen on phones, various widths and sizes expressed in pixels, use of hover attributes for important information like CM labels, etc)

- Perhaps a stretch, but it would be nice to allow some interactivity for checkpoints. Perhaps when you click on a checkpoint, it could open an interstitial window with some prompt and a text box to type a response which could be processed and lead to other prompts (essentially a chatbot). Various checkpoints could be recorded to your scoresheet depending on this back-and-forth. Figuring out how to write the config for this sounds crazy, but maybe there's an off-the-shelf chatbot that could be adapted? Would also need to allow for checkpoints to be included in score rules.
