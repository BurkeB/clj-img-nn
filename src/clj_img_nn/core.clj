(ns clj-img-nn.core
  (:use [mikera.image.core]
        [clj-ml.classifiers] 
        [clj-ml.data] 
        [clj-ml.utils])
  (:require [clojure.java.io :as io]
            [mikera.image.filters :as filt]
            [mikera.image.colours :as c])
  (:gen-class))



;; load an image from a resource file
(def ant (load-image (.getPath (-> "jan-steen.jpg" io/resource))))
(def pixels (get-pixels ant))
(def h (height ant))
(def w (width ant))

(defn round2
  "Round a double to the given precision (number of significant digits)"
  [precision d]
  (let [factor (Math/pow 10 precision)]
    (/ (Math/round (* d factor)) factor)))

(defn get-pos [i]
  (map int [(Math/floor (/ (double  i) (double w)))
             (mod i w)]))

(defn extract-data 
  ([img h w]
   (extract-data (get-pixels img) h w 0 []))
  ([pixels h w i data]
   (if (>= i (* h w))
     data
     (recur pixels h w (inc i) (conj data (flatten [(map #(double (/ %1 %2)) (get-pos i) [h w]) (c/values-rgb (get pixels i))])))
     )))

(println "Extract Data for Image 1")
(def data (filter #(do % (< (rand-int 5) 4)) (extract-data ant h w)))

;; load a second image from a resource file
(def ant (load-image (.getPath (-> "road.jpg" io/resource))))
(def pixels (get-pixels ant))
(def h (height ant))
(def w (width ant))
;; merge both data
(println "Extract Data for Image 2")
(def data (concat data (filter #(do % (< (rand-int 5) 4)) (extract-data ant h w))))


;; create RGB Datasets
(println "Create RGB Datasets")
(def dsR (-> (make-dataset "colorpoints" [:y :x :r :g :b]
                           data)
             (dataset-remove-attribute-at 3)
             (dataset-remove-attribute-at 3)
             (dataset-set-class :r)))

(def dsG (-> (make-dataset "colorpoints" [:y :x :r :g :b]
                           data)
             (dataset-remove-attribute-at 2)
             (dataset-remove-attribute-at 3)
             (dataset-set-class :g)))

(def dsB (-> (make-dataset "colorpoints" [:y :x :r :g :b]
                           data)
             (dataset-remove-attribute-at 2)
             (dataset-remove-attribute-at 2)
             (dataset-set-class :b)))

;; train networks one for each color-channel
(do 
	(println "Train Neural Networks")
  (def cR (-> (make-classifier :neural-network :multilayer-perceptron :hidden-layers-string "2,8,24" :epochs 10000000 :no-nomalization true)
              (classifier-train dsR)))
  (def cG (-> (make-classifier :neural-network :multilayer-perceptron :hidden-layers-string "2,8,24" :epochs 10000000 :no-nomalization true)
              (classifier-train dsG)))
  (def cB (-> (make-classifier :neural-network :multilayer-perceptron :hidden-layers-string "2,8,24" :epochs 10000000 :no-nomalization true)
              (classifier-train dsB))))



(defn pos-to-rgb [y x]
  [(classifier-predict-numeric cR (make-instance dsR {:y y :x x}))
   (classifier-predict-numeric cG (make-instance dsG {:y y :x x}))
   (classifier-predict-numeric cB (make-instance dsB {:y y :x x}))])

(def pixels (get-pixels ant))

(defn rand-element [elements]  
  (nth elements (rand-int 2)))

(defn change-pixels! [pixels i]
  (let [pos (get-pos i)
        rgb-vals (apply pos-to-rgb (map #(round2 7 (/ %1 %2)) pos [h w]))]
    (set-pixel ant (second pos) (first pos) (apply c/rgb rgb-vals))
    (if (>= i (dec (count pixels)))
      pixels
      (recur pixels (inc i)))))

(defn -main
  "I don't do a whole lot ... yet."
  [& args]
  (println "Generate Neural Network Image")
  (change-pixels! pixels 1)
  (show ant)
  (println "Hello, World!"))
