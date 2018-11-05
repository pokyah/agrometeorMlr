# load the required libraries
library(dplyr)
library(sf)
# load the pre-existing datasets
#devtools::install_github("pokyah/agrometVars")
library(agrometeorVars)
# installing new version of agrometAPi
#devtools::install_github("pokyah/agrometAPI")
library(agrometAPI)
library(leaflet)
library(leaflet.extras)

# Read data from the API exported json file downloaded from PAMESEB FTP & create a dataframe
records = jsonlite::fromJSON(
  "./inst/extdata/AGROMET/cleandataSensorstsa-ensForallFm2015-11-11To2018-06-30.json") # available on AGROMET FTP
records.data = records$results
records.meta = records$references$stations
records.l <- list(metadata = records.meta, data = records.data)
records.data <- agrometAPI::type_data(records.l, "cleandata")

# Filtering records to keep only the useful ones (removing non relevant stations)
records.data = records.data %>%
  filter(network_name == "pameseb") %>%
  filter(type_name != "Sencrop") %>%
  filter(!is.na(to)) %>%
  filter(state == "Ok") %>%
  filter(!is.na(tsa)) %>%
  filter(!is.na(ens))

# keeping only sid for joining and the target var
records.data = records.data %>%
  dplyr::select(one_of(c("sid", "mtime", "tsa", "ens")))
# removing temporary vars
rm(records, records.l, records.meta)
# getting only one hour dataset
records.1h.data = records.data %>%
  dplyr::filter(mtime == unique(records.data["mtime"])[1,])

# Read the INCA spatial previsions exported data for November 2015.
# data are stored in the df data.all
load("./inst/extdata/INCABE/INCA_TT_201511.Rdata")
inca.2015.11 = data.all
rm(data.all)

# package agrometVars contains the location info of each INCA px id. Stored in inca object
inca.sf = sf::st_as_sf(inca)
# extracting inca coords as a df
inca.coords = data.frame(sf::st_coordinates(inca.sf))
# extracting px id as a df
inca.px = data.frame(inca.sf$px)
# making a dataframe
inca.df = dplyr::bind_cols(inca.px, inca.coords)
colnames(inca.df) = c("px", "X", "Y")
# joining the locations attributes to the records
inca.2015.11 = inca.2015.11 %>%
  dplyr::left_join(inca.df, by = "px")
# our records.1h.data = 2015-11-11 00:00 UTC. Let's filter inca with this
inca.2015.11.11.1h = inca.2015.11 %>%
  dplyr::filter(DAY == "20151111") %>%
  dplyr::filter(HOUR == 0)

# loading INCA prediction grid extraction
data(inca.ext)
class(inca.ext)

# loading the stations extractions
data(stations.ext)
class(stations.ext)

# transforming geometry col to coords
coords = data.frame(sf::st_coordinates(stations.ext))
sf::st_geometry(stations.ext) = NULL
stations.ext = stations.ext %>%
  dplyr::bind_cols(coords)
coords = data.frame(sf::st_coordinates(inca.ext))
sf::st_geometry(inca.ext) = NULL
inca.ext = inca.ext %>%
  dplyr::bind_cols(coords)
# keeping only useful cols
stations.ext = stations.ext %>%
  dplyr::select(one_of(c(
    "sid", "altitude", "elevation", "slope", "aspect", "Agricultural_areas", "Herbaceous_vegetation", "Forest", "Artificials_surfaces", "X", "Y")))
inca.ext = inca.ext %>%
  dplyr::select(one_of(c(
    "px", "elevation", "slope", "aspect", "Agricultural_areas", "Herbaceous_vegetation", "Forest", "Artificials_surfaces", "X", "Y"))) %>%
  dplyr::select(-px)
# joining the explanatory vars and the target vars and removing mtime + other non explanatory vars
records.1h.data = records.1h.data %>%
  dplyr::left_join(stations.ext, by = "sid") %>%
  dplyr::select(-mtime) %>%
  dplyr::select(-ens) %>%
  dplyr::select(-altitude)
# build the ML task and do not keep sid as it is not a var
# devtools::install_github("pokyah/mlr", ref = "gstat")
# https://www.sciencedirect.com/science/article/pii/S2211675315000482
# https://stackoverflow.com/questions/40527442/r-mlr-wrapper-feature-selection-hyperparameter-tuning-without-nested-nested
# https://github.com/mlr-org/mlr/issues/1861
# https://mlr.mlr-org.com/articles/tutorial/handling_of_spatial_data.html
# https://www.youtube.com/watch?v=LpOsxBeggM0
set.seed(2585)
library(mlr)
records = records.1h.data[-1]
coordinates = records %>%
  dplyr::select(one_of(c("X","Y")))
records = records %>%
  dplyr::select(-one_of("X", "Y"))
regr.task = makeRegrTask(id = "1h", data = records, target = "tsa", coordinates = coordinates)
# grid search for param tuning
ctrl = makeTuneControlGrid()
# absolute number feature selection paramset for fusing learner with the filter method
ps = makeParamSet(makeDiscreteParam("fw.abs", values = seq_len(getTaskNFeats(regr.task))))
# inner spatial resampling loop
# inner = makeResampleDesc("CV", iter = 7)
inner = makeResampleDesc("SpRepCV", fold = 5, reps = 5)
# Outer resampling loop
# outer = makeResampleDesc("CV", inter = 2)
outer = makeResampleDesc("SpRepCV", fold = 5, reps = 5)
# regr.lm learner features filtered with fw.abs param tuned
lrn.lm = makeLearner("regr.lm", predict.type = 'se')
lrn.lm = makeFilterWrapper(learner = lrn.lm, fw.method = "chi.squared")
lrn.lm = makeTuneWrapper(measures = list(rmse, mse), lrn.lm, resampling = inner, par.set = ps, control = ctrl,
  show.info = FALSE)
# regr.glm
lrn.glm = makeLearner("regr.glm", predict.type = 'se')
lrn.glm = makeFilterWrapper(learner = lrn.glm, fw.method = "chi.squared")
lrn.glm = makeTuneWrapper(measures = list(rmse, mse), lrn.glm, resampling = inner, par.set = ps, control = ctrl,
  show.info = FALSE)
# regr.blm
lrn.blm = makeLearner("regr.blm", predict.type = 'se')
lrn.blm = makeFilterWrapper(learner = lrn.blm, fw.method = "chi.squared")
lrn.blm = makeTuneWrapper(measures = list(rmse, mse), lrn.blm, resampling = inner, par.set = ps, control = ctrl,
  show.info = FALSE)
# regr.fnn learner features filtered with fw.abs param tuned
lrn.fnn = makeFilterWrapper(learner = "regr.fnn", fw.method = "chi.squared")
lrn.fnn = makeTuneWrapper(measures = list(rmse, mse), lrn.fnn, resampling = inner, par.set = ps, control = ctrl,
  show.info = FALSE)
# regr.kknn learner features filtered with fw.abs param tuned + k param also tuned
lrn.kknn = makeFilterWrapper(learner = "regr.kknn", fw.method = "chi.squared")
ps.kknn = makeParamSet(
  makeDiscreteParam("fw.abs", values = seq_len(getTaskNFeats(regr.task))),
  makeDiscreteParam("k", c(1,2,3,4,5))
)
lrn.kknn = makeTuneWrapper(measures = list(rmse, mse),lrn.kknn, resampling = inner, par.set = ps.kknn, control = ctrl,
  show.info = FALSE)
# regr.km learner features filtered with fw.abs param tuned + k param also tuned
lrn.km = makeFilterWrapper(learner = "regr.km", fw.method = "chi.squared")
ps.km = makeParamSet(
  makeDiscreteParam("fw.abs", values = seq_len(getTaskNFeats(regr.task)))
)
lrn.km = makeTuneWrapper(measures = list(rmse, mse), lrn.km, resampling = inner, par.set = ps.km, control = ctrl,
  show.info = FALSE)
# regr.gstat learner features filtered with fw.abs param tuned + k param also tuned
lrn.gstat = makeFilterWrapper(learner = "regr.gstat", fw.method = "chi.squared")
ps.gstat = makeParamSet(
  makeDiscreteParam("fw.abs", values = seq_len(getTaskNFeats(regr.task)))
)
lrn.gstat = makeTuneWrapper(measures = list(rmse, mse), lrn.gstat, resampling = inner, par.set = ps.gstat, control = ctrl,
  show.info = FALSE)
# nnet learner features filtered with fw.abs param tuned + size param also tuned
# lrn.nnet = makeFilterWrapper(learner = "regr.nnet", fw.method = "chi.squared")
# ps.nnet = makeParamSet(
#   makeDiscreteParam("fw.abs", values = seq_len(getTaskNFeats(regr.task))),
#   makeDiscreteParam("size", c(1,2,5,10,20,50,80))
# )
# lrn.nnet = makeTuneWrapper(lrn.nnet, resampling = inner, par.set = ps.nnet, control = ctrl,
#   show.info = FALSE)
# cubist learner features filtered with fw.abs param tuned + neighbors param also tuned
lrn.cubist = makeFilterWrapper(learner = "regr.cubist", fw.method = "chi.squared")
ps.cubist = makeParamSet(
  makeDiscreteParam("fw.abs", values = seq_len(getTaskNFeats(regr.task))),
  makeDiscreteParam("neighbors", c(1,2,3,4,5))
)
lrn.cubist = makeTuneWrapper(lrn.cubist, resampling = inner, par.set = ps.cubist, control = ctrl,
  show.info = FALSE)
# Learners
lrns = list(lrn.glm, lrn.fnn, lrn.blm)
# outer = makeResampleDesc("Subsample", iter = 3)
res = benchmark(measures = list(rmse, timetrain), tasks = regr.task, learners = lrns, resamplings = outer, show.info = FALSE)
# nnet
# lrn.nnet = makeLearner("regr.nnet")
# ps.nnet = makeParamSet(
#   makeDiscreteParam("size", c(1,5,10,20))
# )
# lrn.nnet <- makeTuneWrapper(lrn.nnet,
#   resampling = cv3,
#   par.set = ps.nnet,
#   control = ctrl, show.info = FALSE)
# lrn.nnet = makeFeatSelWrapper(lrn.nnet,
#   resampling = cv3,
#   control = makeFeatSelControlSequential(method = "sbs"), show.info = FALSE)

# performances + best learner
perfs = getBMRAggrPerformances(bmr = res, as.df = TRUE)
best.learner = perfs %>%
    slice(which.min(rmse.test.rmse))
# forcing the best learner
best.learner = perfs %>%
  dplyr::filter(learner.id == "regr.fnn.filtered.tuned")
# making quick plots
plotBMRBoxplots(bmr = res, measure = rmse, order.lrn = getBMRLearnerIds(res))
plotBMRSummary(bmr = res)

# training
m = mlr::train(
  learner = mlr::getBMRLearners(bmr = res)[[as.character(best.learner$learner.id)]],
  task = regr.task)

# prediction
p = predict(m, newdata = inca.ext)
s = inca.ext %>%
  dplyr::bind_cols(p$data)

# preparing the required spatial objects for mapping
data("wallonia")
ourPredictedGrid = sf::st_as_sf(s, coords = c("X", "Y"))
ourPredictedGrid = sf::st_set_crs(ourPredictedGrid, 4326)
sfgrid = sf::st_sf(sf::st_make_grid(x = sf::st_transform(wallonia, 3812),  cellsize = 1000, what = "polygons"))
ourPredictedGrid = sf::st_transform(ourPredictedGrid, crs = 3812)
ourPredictedGrid = sf::st_join(sfgrid, ourPredictedGrid)
# limit it to Wallonia
ourPredictedGrid = sf::st_intersection(ourPredictedGrid, sf::st_transform(wallonia, crs = 3812))
# stations
records.sf = sf::st_as_sf(records.1h.data, coords = c("X", "Y"))
records.sf = sf::st_set_crs(records.sf, 4326)


# Definition of the function to build a leaflet map for prediction with associated uncertainty
leafletize <- function(data.sf, borders, stations){

  # be sure we are in the proper 4326 EPSG
  data.sf = sf::st_transform(data.sf, 4326)
  stations = sf::st_transform(stations, 4326)

  # to make the map responsive
  responsiveness.chr = "\'<meta name=\"viewport\" content=\"width=device-width, initial-scale=1.0\">\'"

  # Sometimes the interpolation and the stations don't have values in the same domain.
  # this lead to mapping inconsistency (transparent color for stations)
  # Thus we create a fullDomain which is a rowbinding of interpolated and original data
  fullDomain = c(data.sf$response, stations$tsa)

  # defining the color palette for the response
  varPal <- leaflet::colorNumeric(
    palette = "RdYlBu", #"RdBl",
    reverse = TRUE,
    domain = fullDomain, #data.sf$response,
    na.color = "transparent"
  )

  # Definition of the function to create whitening
  alphaPal <- function(color) {
    alpha <- seq(0,1,0.1)
    r <- col2rgb(color, alpha = T)
    r <- t(apply(r, 1, rep, length(alpha)))
    # Apply alpha
    r[4,] <- alpha*255
    r <- r/255.0
    codes <- (rgb(r[1,], r[2,], r[3,], r[4,]))
    return(codes)
  }

  # actually building the map
  prediction.map = leaflet::leaflet(data.sf) %>%
    # basemaps
    addProviderTiles(group = "Stamen",
      providers$Stamen.Toner,
      options = providerTileOptions(opacity = 0.25)
    ) %>%
    addProviderTiles(group = "Satellite",
      providers$Esri.WorldImagery,
      options = providerTileOptions(opacity = 1)
    ) %>%
    # centering the map
    fitBounds(sf::st_bbox(data.sf)[[1]],
      sf::st_bbox(data.sf)[[2]],
      sf::st_bbox(data.sf)[[3]],
      sf::st_bbox(data.sf)[[4]]
    ) %>%
    # adding layer control button
    addLayersControl(baseGroups = c("Stamen", "Satellite"),
      overlayGroups = c("prediction", "se", "Stations", "Admin"),
      options = layersControlOptions(collapsed = TRUE)
    ) %>%
    # fullscreen button
    addFullscreenControl() %>%
    # location button
    addEasyButton(easyButton(
      icon = "fa-crosshairs", title = "Locate Me",
      onClick = JS("function(btn, map){ map.locate({setView: true}); }"))) %>%
    htmlwidgets::onRender(paste0("
      function(el, x) {
      $('head').append(",responsiveness.chr,");
      }")
    ) %>%
    # predictions
    addPolygons(
      group = "prediction",
      color = "#444444", stroke = FALSE, weight = 1, smoothFactor = 0.8,
      opacity = 1.0, fillOpacity = 0.9,
      fillColor = ~varPal(response),
      highlightOptions = highlightOptions(color = "white", weight = 2,
        bringToFront = TRUE),
      label = ~htmltools::htmlEscape(as.character(response))
    ) %>%
    addLegend(
      position = "bottomright", pal = varPal, values = ~response,
      title = "prediction",
      group = "prediction",
      opacity = 1
    )

  # if se.bool = TRUE
  if (!is.null(data.sf$se)) {
    uncPal <- leaflet::colorNumeric(
      palette = alphaPal("#e6e6e6"),
      domain = data.sf$se,
      alpha = TRUE
    )

    prediction.map = prediction.map %>%
      addPolygons(
        group = "se",
        color = "#444444", stroke = FALSE, weight = 1, smoothFactor = 0.5,
        opacity = 1.0, fillOpacity = 1,
        fillColor = ~uncPal(se),
        highlightOptions = highlightOptions(color = "white", weight = 2,
          bringToFront = TRUE),
        label = ~ paste("prediction:", signif(data.sf$response, 2), "\n","se: ", signif(data.sf$se, 2))
      ) %>%
      addLegend(
        group = "se",
        position = "bottomleft", pal = uncPal, values = ~se,
        title = "se",
        opacity = 1
      )
  }

  prediction.map = prediction.map %>%
    # admin boundaries
    addPolygons(
      data = borders,
      group = "Admin",
      color = "#444444", weight = 1, smoothFactor = 0.5,
      opacity = 1, fillOpacity = 0, fillColor = FALSE) %>%
    # stations location
    addCircleMarkers(
      data = stations,
      group = "Stations",
      color = "black",
      weight = 2,
      fillColor = ~varPal(tsa),
      stroke = TRUE,
      fillOpacity = 1,
      label = ~htmltools::htmlEscape(as.character(tsa)))

  return(prediction.map)
}

# creating the interactive map
predicted_map = leafletize(
  data.sf = ourPredictedGrid,
  borders = wallonia,
  stations = records.sf
)
predicted_map


