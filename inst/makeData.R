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
# feature selection paramset for fusing
ps = makeParamSet(makeDiscreteParam("fw.abs", values = seq_len(getTaskNFeats(regr.task))))
# inner resamling loop
# inner = makeResampleDesc("CV", iter = 7)
inner = makeResampleDesc("SpRepCV", fold = 5, reps = 5)
# Outer resampling loop
# outer = makeResampleDesc("CV", inter = 2)
outer = makeResampleDesc("SpRepCV", fold = 5, reps = 5)
# regr.lm learner features filtered and param tuned
lrn.lm = makeFilterWrapper(learner = "regr.lm", fw.method = "chi.squared")
lrn.lm = makeTuneWrapper(lrn.lm, resampling = inner, par.set = ps, control = ctrl,
  show.info = FALSE)
# regr.glm
lrn.glm = makeFilterWrapper(learner = "regr.glm", fw.method = "chi.squared")
lrn.glm = makeTuneWrapper(lrn.glm, resampling = inner, par.set = ps, control = ctrl,
  show.info = FALSE)
# regr.fnn learner features filtered and param tuned
lrn.fnn = makeFilterWrapper(learner = "regr.fnn", fw.method = "chi.squared")
lrn.fnn = makeTuneWrapper(lrn.fnn, resampling = inner, par.set = ps, control = ctrl,
  show.info = FALSE)
# regr.kknn learner features filtered and param tuned
lrn.kknn = makeFilterWrapper(learner = "regr.kknn", fw.method = "chi.squared")
ps.kknn = makeParamSet(
  makeDiscreteParam("fw.abs", values = seq_len(getTaskNFeats(regr.task))),
  makeDiscreteParam("k", c(1,2,3,4,5))
)
lrn.kknn = makeTuneWrapper(lrn.kknn, resampling = inner, par.set = ps, control = ctrl,
  show.info = FALSE)
# nnet
lrn.nnet = makeFilterWrapper(learner = "regr.nnet", fw.method = "chi.squared")
ps.nnet = makeParamSet(
  makeDiscreteParam("fw.abs", values = seq_len(getTaskNFeats(regr.task))),
  makeDiscreteParam("size", c(1,2,5,10,20,50))
)
lrn.nnet = makeTuneWrapper(lrn.nnet, resampling = inner, par.set = ps.nnet, control = ctrl,
  show.info = FALSE)
# cubist
lrn.cubist = makeFilterWrapper(learner = "regr.cubist", fw.method = "chi.squared")
lrn.cubist = makeTuneWrapper(lrn.cubist, resampling = inner, par.set = ps, control = ctrl,
  show.info = FALSE)
# Learners
lrns = list("regr.lm", lrn.lm, lrn.nnet, lrn.fnn, lrn.kknn, lrn.cubist, lrn.glm)
# outer = makeResampleDesc("Subsample", iter = 3)
res = benchmark(measures = rmse, tasks = regr.task, learners = lrns, resampling = outer, show.info = FALSE)
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
