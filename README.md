# Team Waypoint to Mensa: Makeathon 2025 Submission

 *Team Members: Vincent Limbach, Damian Hattler, Menalaos Fotiadis, Kesava Prasad Arul*

![Architecture](assets/arch.jpg)

### Developer Notes:
* The repository consists of several services, as depicted in the picture above.
* Each service has their own Docker file and are also hosted here:
    * Dataset Engine: [https://dataset.kesava.lol](https://dataset.kesava.lol)
    * Open Drone Map: [https://opendrone.kesava.lol](https://opendrone.kesava.lol)
    * Segmentation Engine: [https://point_api.kesava.lol](https://point_api.kesava.lol)
    * Pixel to Geopoint Translation Engine: [https://pixeltranslate.kesava.lol](https://pixeltranslate.kesava.lol)
    * KML Generator: [https://kmlgen.kesava.lol](https://kmlgen.kesava.lol)

Each folder has their own organizational structure, although fairly straight forward. All services are hosted via CloudFlare, and DockerHub for Image Management. All services are also supported with Swagger OpenAPI support and ReDoc containers. Visit the website (with /docs for swagger) for more API endpoint documentation.

The larger files - original datasets, segmentation models, pre-processed static data, processed drone data, reconstruction models, calibration and error models are present in [In OneDrive](https://1drv.ms/f/c/96185f2c475d5289/Eo9ZhlM2xvtAogXkV7gwnuUBhjPQzpYdKokt3IsP3KGNxg?e=lt3n5L)(deleted in 45 days)


