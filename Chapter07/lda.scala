import org.apache.spark.graphx._
import org.apache.spark.mllib.linalg._
import org.apache.spark.mllib.clustering._

import org.apache.spark.rdd._

def bagsFromDocumentPerLine(filename: String) = sc.textFile(filename).map(_.split(' ').filter(
        x => x.length > 5 && x.toLowerCase != "reuter").map(_.toLowerCase).groupBy(
        x => x).toList.map(x => (x._1, x._2.size)))
/*
// groupBy 用法示例
scala> val s = "Viacom International Inc said ltNational Amusements Inc has again raised the value of its offer for Viacoms publicly held stock     The company said the special committee of its board plans to meet later today to consider this offer and the one submitted March one by ltMCV Holdings Inc     A spokeswoman was unable to say if the committee met as planned yesterday     Viacom said National Amusements Arsenal Holdings Inc subsidiary has raised the amount of cash it is offering for each Viacom share by  cts to  dlrs while the value of the fraction of a share of exchangeable Arsenal Holdings preferred to be included was raised  cts to  dlrs     National Amusements already owns  pct of Viacoms stock  Reuter nTEXT REUTERS"
scala> s.split(' ').filter(x => x.length > 5 && x.toLowerCase != "reuter").map(_.toLowerCase).groupBy(x => x)
res3: scala.collection.immutable.Map[String,Array[String]] = Map(spokeswoman -> Array(spokeswoman), submitted -> Array(submitted), fraction -> Array(fraction), arsenal -> Array(arsenal, arsenal), subsidiary -> Array(subsidiary), already -> Array(already), consider -> Array(consider), included -> Array(included), preferred -> Array(preferred), viacoms -> Array(viacoms, viacoms), unable -> Array(unable), publicly -> Array(publicly), exchangeable -> Array(exchangeable), viacom -> Array(viacom, viacom, viacom), committee -> Array(committee, committee), amount -> Array(amount), national -> Array(national, national), company -> Array(company), reuters -> Array(reuters), yesterday -> Array(yesterday), planned -> Array(planned), international -> Array(international), offering -> Array(offering)...
scala> s.split(' ').filter(x => x.length > 5 && x.toLowerCase != "reuter").map(_.toLowerCase).groupBy(x => x).toList.map(x => (x._1, x._2.size))
res6: List[(String, Int)] = List((spokeswoman,1), (submitted,1), (fraction,1), (arsenal,2), (subsidiary,1), (already,1), (consider,1), (included,1), (preferred,1), (viacoms,2), (unable,1), (publicly,1), (exchangeable,1), (viacom,3), (committee,2), (amount,1), (national,2), (company,1), (reuters,1), (yesterday,1), (planned,1), (international,1), (offering,1), (raised,3), (amusements,3), (holdings,3), (special,1), (ltnational,1))
 */

val rddBags: RDD[List[Tuple2[String, Int]]] = bagsFromDocumentPerLine("/app/dt/udw/user/wenjurong/spark/input/rcorpus")

def vocab: Array[Tuple2[String, Long]] = rddBags.flatMap(x => x).reduceByKey(_ + _).map(_._1).zipWithIndex.collect
/*
// flatMap、zipWithIndex 示例
scala> rddBags.flatMap(x => x).collect
res12: Array[(String, Int)] = Array((spokeswoman,1), (submitted,1), (fraction,1), (arsenal,2), (subsidiary,1), (already,1), (consider,1), (included,1), (preferred,1), (viacoms,2), (unable,1), (publicly,1), (exchangeable,1), (viacom,3), (committee,2), (amount,1), (national,2), (company,1), (reuters,1), (yesterday,1), (planned,1), (international,1), (offering,1), (raised,3), (amusements,3), (holdings,3), (special,1), (ltnational,1), (number,1), (people,1), (benefits,2), (applications,1), (insurance,1), (unemployment,1), (previous,1), (adjusted,1), (regular,1), (programs,1), (department,1), (reuters,1), (actually,1), (seasonally,1), (receiving,1), (available,1), (totaled,1), (latest,1), (figure,1), (period,1), (luxembourg,1), (denominations,1), (concession,1), (coupon,1), (outstanding,1), ...
scala> rddBags.flatMap(x => x).reduceByKey(_ + _).map(_._1).zipWithIndex.collect
res17: Array[(String, Long)] = Array((package,0), (expects,1), (country,2), (cautiously,3), (improvement,4), (intervals,5), (extraordinary,6), (strengthening,7), (expirations,8), (produced,9), (exchanges,10), (signup,11), (longterm,12), (include,13), (president,14), (national,15), (venice,16), (appreciations,17), (reserve,18), (export,19), (however,20), (salaries,21), (santana,22), (baldrige,23), (movements,24), (combining,25), (predicted,26), (largely,27), (implementation,28), (select,29), (instead,30), (offering,31), (connection,32), (immediately,33), (amount,34), (halted,35), (industrialised,36), (presented,37), (uncertainties,38), (western,39), (intention,40), (exchequer,41), (yeartodate,42), (consumer,43), (marketing,44), (doctrine,45), (urgently,46), (economy,47), (accord,48), (am...
 */

def codeBags(rddBags: RDD[List[Tuple2[String, Int]]])=
rddBags.map(x => (x ++ vocab).groupBy(_._1).filter(_._2.size > 1).map(
        x => (x._2(1)._2.asInstanceOf[Long].toInt,
                x._2(0)._2.asInstanceOf[Long].toDouble)).toList
).zipWithIndex.map(x => (x._2, new SparseVector(
    vocab.size,
    x._1.map(_._1).toArray,
    x._1.map(_._2).toArray
).asInstanceOf[Vector]))

val model = new LDA().setK(5).run(codeBags(rddBags))
