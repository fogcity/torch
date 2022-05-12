import React, { useEffect, useState } from "react";
import { buildNetwork, forwardProp } from "../lib/nn";
import { mse } from "../lib/loss";
import { sgd } from "../lib/optim";
import { Activation } from "../lib/activation";
import { vis } from "../../vis/example/build";
import * as d3 from "d3";
export const NN = () => {
  useEffect(() => {
    // import data,labels & handle them
    d3.csv("/data/fashion-mnist_train.csv").then((v) => {
      console.time("t");
      const featureNumber = 2;
      const dataNumber = 1;
      let datas = v
        .slice(0, dataNumber)
        .map((r) =>
          Object.values(r).slice(0, featureNumber + 1)
        ) as unknown as number[][];

      const labels = datas.map((d) => d[0]);
      datas = datas.map((d) => d.slice(1, d.length));

      // create network model
      const model = buildNetwork([featureNumber, 2, 1], Activation.SIGMOID);
      // set learning rate
      const lr = 0.001;
      // set regular rate
      const rr = 0.01;
      // set mini-batch size (update weight and bias per how many times backward)
      const batchSize = 1;
      // train how many times
      const epoch = 2;
      // create an optimizer by sgd
      const optimizer = sgd(model, lr, rr);
      // create an loss function by mse
      const loss = mse(model, datas, labels);
      const lossValuePoints: number[] = [];
      // begin training
      for (let i = 0; i < epoch; i++) {
        datas.forEach((d, i) => {
          // do once  forwardprop to get one data's output
          forwardProp(model, d);
          // do once backprop
          loss.backProp();
          // once during a batch
          if ((i + 1) % batchSize == 0) {
            // use optimizer update weights & bias
            optimizer.step();
            // record loss value to draw chart after
            lossValuePoints.push(loss());
          }
        });
      }
      console.log("lossPoints", lossValuePoints);
      console.log("datas", datas);
      console.log("labels", labels);
      //   setLossPoints(lossPoints.map((v,i)=>[i,v]));
    });
  }, []);

  //   useEffect(() => {
  //     if (lossPoints.length > 0) {
  //       console.timeEnd("t");
  //       vis.renderLineChart(document.getElementById("root"), lossPoints, {
  //         height: 400,
  //         width: 400,
  //         xLabel: "epoch",
  //         yLabel: "loss",
  //         marginLeft: 50,
  //         marginBottom: 50,
  //         showXAxisGrid: true,
  //         showYAxisGrid: true,
  //         color: "rgb(107, 0, 255)",
  //         lineWidth: 2.5,
  //         curve: d3.curveNatural,
  //         xAccessor: (d) => d[0],
  //         yAccessor: (d) => d[1],
  //       });
  //     }
  //   }, [lossPoints]);

  return <></>;
};
