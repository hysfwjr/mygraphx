import org.apache.spark.graphx._

val edges = sc.makeRDD(Array(
    Edge(1L, 11L, 5.0), Edge(1L, 12L, 4.0), Edge(2L, 12L, 5.0),
    Edge(2L, 13L, 5.0), Edge(3L, 11L, 5.0), Edge(3L, 13L, 2.0),
    Edge(4L, 11L, 4.0), Edge(4L, 12L, 4.0)
))

val conf = new lib.SVDPlusPlus.Conf(2, 10, 0, 5, 0.007, 0.007, 0.005, 0.015)
val (g, mean) = lib.SVDPlusPlus.run(edges, conf)

// 预测
def pred(g: Graph[(Array[Double], Array[Double], Double, Double), Double],
        mean:Double, u:Long, i:Long) = {
    val user = g.vertices.filter(_._1 == 4L).collect()(0)._2
    val item = g.vertices.filter(_._1 == 13L).collect()(0)._2
    mean + user._3 + item._3 + item._1.zip(user._2).map(
            x => x._1 * x._2).reduce(_ + _)
}
pred(g, mean, 4L, 13L)
