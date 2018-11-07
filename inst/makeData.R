# build the ML task and do not keep sid as it is not a var
# devtools::install_github("pokyah/mlr", ref = "gstat")
# https://www.sciencedirect.com/science/article/pii/S2211675315000482
# https://stackoverflow.com/questions/40527442/r-mlr-wrapper-feature-selection-hyperparameter-tuning-without-nested-nested
# https://github.com/mlr-org/mlr/issues/1861
# https://mlr.mlr-org.com/articles/tutorial/handling_of_spatial_data.html
# https://www.youtube.com/watch?v=LpOsxBeggM0

# load the required libraries
library(dplyr)
# devtools::install_github("r-spatial/sf")
library(sf)
# devtools::install_github("pokyah/agrometVars")
library(agrometeorVars)
# installing new version of agrometAPI
#devtools::install_github("pokyah/agrometAPI")
library(agrometAPI)
# installing custo mversion of mlr
#devtools::install_github("pokyah/mlr", ref="gstat")
library(mlr)
library(leaflet)
library(leaflet.extras)

# loading the datasets from agrometeorVars package
data("grid.sf")
data("grid.static")
data("grid.dyn")

data("stations.sf")
data("stations.static")
data("stations.dyn")

data("intersections")

# defining test.dateTime object for testing purpose
test.dateTime = stations.dyn$mtime[2]

# defining the function that create a task for a specific hour (mtime)
make.1h.data = function(dateTime){
  # extracting the tsa_hp1 var at stations points for current mtime
  stations.mtime = stations.dyn %>%
    # filtering for current time
    dplyr::filter(mtime == dateTime) %>%
    # adding corresponding grid px
    dplyr::left_join(
      data.frame(stations.sf)[c("sid", "px")],
      by = "sid"
    ) %>%
    # adding tsa_hp1
    dplyr::left_join(
      (grid.dyn %>%
          dplyr::filter(mtime == dateTime) %>%
          dplyr::select(c("px", "tsa_hp1"))
      ),
      by = "px"
    ) %>%
    # adding static vars
    dplyr::left_join(
      stations.static,
      by = "sid"
    ) %>%
    # removing useless px
    dplyr::select(-px) %>%
    # adding the lat and lon as projected Lambert 2008 (EPSG = 3812)
    dplyr::left_join(
      (data.frame(st_coordinates(st_transform(stations.sf, 3812))) %>%
        dplyr::bind_cols(stations.sf["sid"]) %>%
        dplyr::select(-geometry)
      ),
      by = "sid"
    ) %>%
    # removing mtime
    dplyr::select(-mtime) %>%
    # renaming X and Y to x and y
    dplyr::rename(x = X) %>%
    dplyr::rename(y = Y)
}

test.1hdata = make.1h.data(dateTime = test.dateTime)

# defining the function that performs the benchmark
make.bmr = function(data, dateTime, learners){
  # make bmr reproducible
  set.seed(2585)
  # defining the task
  task.dateTime = mlr::makeRegrTask(
    id = as.character(dateTime),
    data = data,
    target = "tsa"
  )
  # ::FIXME:: coordinates ?
  # removing ens for now
  task.dateTime = dropFeatures(task.dateTime, "ens")

  # grid search for param tuning
  ctrl = makeTuneControlGrid()

  # absolute number feature selection paramset for fusing learner with the filter method
  ps = makeParamSet(makeDiscreteParam("fw.abs", values = seq_len(getTaskNFeats(task.dateTime))))

  # inner resampling loop
  inner = makeResampleDesc("LOO")

  # outer resampling loop
  outer = makeResampleDesc("LOO")

  # benchmarking
  res = benchmark(
    measures = list(mae, mse, rmse, timetrain),
    tasks = task.dateTime,
    learners = learners,
    resamplings = outer,
    show.info = FALSE
  )

}

# defining the simple learners
lrn.lm.alt_x_y = makeFilterWrapper(
  learner = makeLearner(
    cl = "regr.lm",
    id = "multiReg.alt_x_y",
    predict.type = 'se'),
  fw.method = "linear.correlation",
  fw.mandatory.feat = c("altitude", "y", "x"),
  fw.abs = 3)

lrn.gstat.idw = makeLearner(
  cl = "regr.gstat",
  id = "idw",
  predict.type = "se")

lrn.gstat.ts1 = makeLearner(
  cl = "regr.gstat",
  id = "ts1",
  par.vals = list(degree = 1),
  predict.type = "se")

lrn.gstat.ts2 = makeLearner(
  cl = "regr.gstat",
  id = "ts2",
  par.vals = list(degree = 2),
  predict.type = "se")

lrn.gstat.ok = makeFilterWrapper(
  learner = makeLearner(
    cl = "regr.gstat",
    id = "ok",
    par.vals = list(
      range = 800,
      psill = 2500,
      model.manual = "Sph",
      nugget = 0),
    predict.type = "se"),
  fw.method = "linear.correlation",
  fw.mandatory.feat = c("y", "x"),
  fw.abs = 2)

lrn.gstat.ked = makeFilterWrapper(
  learner = makeLearner(
    cl = "regr.gstat",
    id = "ked",
    par.vals = list(
      range = 800,
      psill = 2500,
      model.manual = "Sph",
      nugget = 0),
    predict.type = "se"),
  fw.method = "linear.correlation",
  fw.mandatory.feat = c("y", "x", "altitude"),
  fw.abs = 3)

# learners in a list
lrns = list(lrn.lm.alt_x_y, lrn.gstat.idw, lrn.gstat.ts1, lrn.gstat.ts2, lrn.gstat.ok, lrn.gstat.ked)

hours = as.list(stations.dyn$mtime)

bmrs = purrr::map(
  .x = hours,
  .f = make.bmr,
  data = make.1h.data,
  learners = lrns)






test.bmr = make.bmr(data = test.1hdata, dateTime = test.dateTime, learners = lrns)
test.bmr
getBMRFilteredFeatures(test.bmr)




regr.task = makeRegrTask(id = "1h", data = records, target = "tsa", coordinates = coordinates)
# grid search for param tuning
ctrl = makeTuneControlGrid()
# absolute number feature selection paramset for fusing learner with the filter method
ps = makeParamSet(makeDiscreteParam("fw.abs", values = seq_len(getTaskNFeats(regr.task))))
# inner spatial resampling loop
# inner = makeResampleDesc("CV", iter = 7)
# inner = makeResampleDesc("SpRepCV", fold = 5, reps = 5)
inner = makeResampleDesc("LOO")
# Outer resampling loop
# outer = makeResampleDesc("CV", inter = 2)
# outer = makeResampleDesc("SpRepCV", fold = 5, reps = 5)
outer = makeResampleDesc("LOO")
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
#lrn.gstat = makeFilterWrapper(learner = "regr.gstat", fw.method = "chi.squared")
# ps.gstat = makeParamSet(
#   makeDiscreteParam("fw.abs", values = seq_len(getTaskNFeats(regr.task)))
# )
# lrn.gstat = makeTuneWrapper(measures = list(rmse, mse), lrn.gstat, resampling = inner, par.set = ps.gstat, control = ctrl,
#   show.info = FALSE)
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
lrns = list(lrn.glm, lrn.lm, lrn.fnn)
# outer = makeResampleDesc("Subsample", iter = 3)
res = benchmark(measures = list(mae, mse, rmse, timetrain), tasks = regr.task, learners = lrns, resamplings = outer, show.info = FALSE)
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
# best.learner = perfs %>%
#   dplyr::filter(learner.id == "regr.fnn.filtered.tuned")
# # making quick plots
# plotBMRBoxplots(bmr = res, measure = rmse, order.lrn = getBMRLearnerIds(res))
# plotBMRSummary(bmr = res)

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
      palette = alphaPal("#5af602"),
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


