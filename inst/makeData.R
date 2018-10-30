# load the required libraries
library(dplyr)
library(sf)
# load the pre-existing datasets
#devtools::install_github("pokyah/agrometVars")
library(agrometeorVars)
# installing new version of agrometAPi
#devtools::install_github("pokyah/agrometAPI")
library(agrometAPI)
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
# loading the stations extraction + prediction grid extraction
data(inca.ext)
class(inca.ext)
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
    "px", "altitude", "elevation", "slope", "aspect", "Agricultural_areas", "Herbaceous_vegetation", "Forest", "Artificials_surfaces", "X", "Y"))) %>%
  dplyr::select(-px)
# joining the explanatory vars and the target vars and removing mtime
records.1h.data = records.1h.data %>%
  dplyr::left_join(stations.ext, by = "sid") %>%
  dplyr::select(-mtime) %>%
  dplyr::select(-ens) %>%
  dplyr::select(-altitude)
# build the ML task and do not keep sid as it is not a var
# devtools::install_github("pokyah/mlr", ref = "gstat")
library(mlr)
regr.task = makeRegrTask(id = "1h", data = records.1h.data[-1], target = "tsa")
# create the learning algo with tuned feature selection
# Tuning of the percentage of selected filters in the inner loop
lrn = makeFilterWrapper(learner = "regr.lm", fw.method = "chi.squared")
ps = makeParamSet(makeDiscreteParam("fw.abs", values = seq_len(getTaskNFeats(regr.task))))
ctrl = makeTuneControlGrid()
inner = makeResampleDesc("LOO") # ::FIXME:: spatial CV
lrn = makeTuneWrapper(measures = rmse, lrn, resampling = inner, par.set = ps, control = ctrl, show.info = FALSE)

# Learners
lrns = list("regr.lm", lrn)

# Outer resampling loop
outer = makeResampleDesc("LOO")
res = benchmark(measures = rmse, tasks = regr.task, learners = lrns, resampling = outer, show.info = FALSE)

r = resample(measures = rmse, learner = lrn, task = regr.task, resampling = outer, models = TRUE, show.info = FALSE)
r$models
lapply(r$models, function(x) getFilteredFeatures(x$learner.model$next.model))
res = lapply(r$models, getTuneResult)
res


# https://github.com/mlr-org/mlr/issues/1861
# https://stackoverflow.com/questions/40527442/r-mlr-wrapper-feature-selection-hyperparameter-tuning-without-nested-nested
# https://mlr.mlr-org.com/articles/tutorial/nested_resampling.html#example-3-one-task-two-learners-feature-filtering-with-tuning

# tuning of the parameters
# Feature filtering with tuning in the inner resampling loop
lrn = makeFilterWrapper(learner = "regr.nnet", fw.method = "chi.squared")
ps = makeParamSet(
  makeDiscreteParam("size", values = c(1, 2, 5, 10, 30, 60, 70))
)
ctrl = makeTuneControlGrid()
inner = makeResampleDesc("CV", iter = 2)
lrn = makeTuneWrapper(lrn, resampling = inner, par.set = ps, control = ctrl,
  show.info = FALSE)

# Learners
lrns = list("regr.nnet", lrn)

# Outer resampling loop
outer = makeResampleDesc("Subsample", iter = 3)
res = benchmark(tasks = regr.task, learners = lrns, resampling = outer, show.info = FALSE)


ps = makeParamSet(
  makeDiscreteParam("size", values = c(1, 2, 5, 10, 30, 60, 70))
)
# grid search:
ctrl = makeTuneControlGrid()
# specify the learner
lrn <- makeLearner("regr.nnet")
# generate a tune wrapper:
lrn <- makeTuneWrapper(lrn, measures = rmse, resampling = cv3, par.set = ps, control = makeTuneControlGrid(), show.info = FALSE)
# generate a feat sel wrapper
lrn = makeFeatSelWrapper(lrn,
  measures = rmse,
  resampling = cv3,
  control = makeFeatSelControlSequential(method = "sbs"), show.info = FALSE)
# perform resampling
res <- resample(lrn, task = regr.task,  resampling = cv3, show.info = TRUE, models = TRUE)
# res = benchmark(measures = rmse, tasks = regr.task, learners = lrns, resampling = outer, show.info = FALSE)

rdesc = makeResampleDesc("LOO")
res = tuneParams(measures = rmse, "regr.nnet", task = regr.task, resampling = rdesc,
  par.set = ps, control = ctrl)
res$x
res$y
lrn = setHyperPars(makeLearner("regr.nnet"), par.vals = res$x)
# create the model
m = train(lrn, regr.task)



# Feature filtering with tuning in the inner resampling loop
lrn = makeFilterWrapper(learner = "regr.lm", fw.method = "chi.squared")
ps = makeParamSet(makeDiscreteParam("fw.abs", values = seq_len(getTaskNFeats(bh.task))))
ctrl = makeTuneControlGrid()
inner = makeResampleDesc("CV", iter = 2)
lrn = makeTuneWrapper(lrn, resampling = inner, par.set = ps, control = ctrl,
  show.info = FALSE)

# Learners
lrns = list("regr.rpart", lrn)

# Outer resampling loop
outer = makeResampleDesc("Subsample", iter = 3)
res = benchmark(measures = rmse, tasks = regr.task, learners = lrns, resampling = outer, show.info = FALSE)

# performances + best learner
perfs = getBMRAggrPerformances(bmr = res, as.df = TRUE)
best.learner = perfs %>%
    slice(which.min(rmse.test.rmse))

# training
m = mlr::train(
  learner = mlr::getBMRLearners(bmr = res)[[as.character(best.learner$learner.id)]],
  task = regr.task)

# prediction
p = predict(m, newdata = inca.ext)
s = inca.ext %>%
  dplyr::bind_cols(p$data)

# map
data("wallonia")
ourPredictedGrid = sf::st_as_sf(s, coords = c("X", "Y"))
ourPredictedGrid = sf::st_set_crs(ourPredictedGrid, 4326)
sfgrid = sf::st_sf(sf::st_make_grid(x = sf::st_transform(wallonia, 3812),  cellsize = 1000, what = "polygons"))
ourPredictedGrid = sf::st_transform(ourPredictedGrid, crs = 3812)
ourPredictedGrid = sf::st_join(sfgrid, ourPredictedGrid)
ourPredictedGrid = sf::st_transform(ourPredictedGrid, 4326)


temperature.pal <- colorNumeric(reverse = TRUE, "RdBu", domain=ourPredictedGrid$response,
  na.color = "transparent")
responsiveness = "\'<meta name=\"viewport\" content=\"width=device-width, initial-scale=1.0\">\'"
map <- leaflet() %>%
  addProviderTiles(
    providers$OpenStreetMap.BlackAndWhite, group = "B&W") %>%
  addProviderTiles(
    providers$Esri.WorldImagery, group = "Satelitte") %>%
  addPolygons(
    data = wallonia, group = "Admin", color = "#444444", weight = 1, smoothFactor = 0.5,
    opacity = 1, fillOpacity = 0.1, fillColor = "grey") %>%
  addPolygons(
    data = ourPredictedGrid,
    group = "Predictions",
    color = ~temperature.pal(response),
    stroke = FALSE,
    fillOpacity = 0.9,
    label = ~htmltools::htmlEscape(as.character(response))) %>%
  addEasyButton(easyButton(
    icon = "fa-crosshairs", title = "Locate Me",
    onClick = JS("function(btn, map){ map.locate({setView: true}); }"))) %>%
  htmlwidgets::onRender(paste0("
       function(el, x) {
       $('head').append(",responsiveness,");
       }"))
