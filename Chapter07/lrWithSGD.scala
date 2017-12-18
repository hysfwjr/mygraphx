import org.apache.spark.graphx._
import org.apache.spark.mllib.classification.LogisticRegressionWithSGD

val trainV = sc.makeRDD(Array((1L, (0,1,false)), (2L, (0,0,false)),
  (3L, (1,0,false)), (4L, (0,0,false)), (5L, (0,0,false)),
  (6L, (0,0,false)), (7L, (0,0,false)), (8L, (0,0,false)),
  (9L, (0,1,false)), (10L,(0,0,false)), (11L,(5,2,true)),
  (12L,(0,0,true)),  (13L,(1,0,false))))

val trainE = sc.makeRDD(Array(Edge(1L,9L,""), Edge(2L,3L,""),
  Edge(3L,10L,""), Edge(4L,9L,""), Edge(4L,10L,""), Edge(5L,6L,""),
  Edge(5L,11L,""), Edge(5L,12L,""), Edge(6L,11L,""), Edge(6L,12L,""),
  Edge(7L,8L,""), Edge(7L,11L,""), Edge(7L,12L,""), Edge(7L,13L,""),
  Edge(8L,11L,""), Edge(8L,12L,""), Edge(8L,13L,""), Edge(9L,2L,""),
  Edge(9L,13L,""), Edge(10L,13L,""), Edge(12L,9L,"")))

val trainG = Graph(trainV, trainE)

import org.apache.spark.graphx.lib.PageRank
import org.apache.spark.mllib.linalg.DenseVector
import org.apache.spark.mllib.regression.LabeledPoint

def augment(g: Graph[Tuple3[Int, Int, Boolean], String]) = g.vertices.join(
    PageRank.run(trainG, 1).vertices.join(
        PageRank.run(trainG, 5).vertices
    ).map(x => (x._1, x._2._2 / x._2._1))
).map(x => LabeledPoint(
    if (x._2._1._3) 1 else 0,
    new DenseVector(Array(x._2._1._1, x._2._1._2, x._2._2))
))

val trainSet = augment(trainG)
val model = LogisticRegressionWithSGD.train(trainSet, 10)

import org.apache.spark.rdd.RDD

def perf(s: RDD[LabeledPoint]) = 100 * (s.count -
    s.map(x => math.abs(model.predict(x.features) - x.label)).reduce(_ + _)
) / s.count

perf(trainSet)

val testV = sc.makeRDD(Array((1L, (0,1,false)), (2L, (0,0,false)),
  (3L, (1,0,false)), (4L, (5,4,true)), (5L, (0,1,false)),
  (6L, (0,0,false)), (7L, (1,1,true))))

val testE = sc.makeRDD(Array(Edge(1L,5L,""), Edge(2L,5L,""),
  Edge(3L,6L,""), Edge(4L,6L,""), Edge(5L,7L,""), Edge(6L,7L,"")))

perf(augment(Graph(testV,testE)))
