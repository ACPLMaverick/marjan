package mainPackage.View;

import java.awt.Dimension;
import java.awt.Paint;
import java.awt.PaintContext;
import java.awt.Rectangle;
import java.awt.RenderingHints;
import java.awt.Transparency;
import java.awt.geom.AffineTransform;
import java.awt.geom.Rectangle2D;
import java.awt.image.ColorModel;
import java.awt.image.Raster;
import java.awt.image.WritableRaster;
import java.util.ArrayList;

import org.jfree.chart.ChartFactory;
import org.jfree.chart.ChartPanel;
import org.jfree.chart.JFreeChart;
import org.jfree.chart.axis.DateAxis;
import org.jfree.chart.plot.PlotOrientation;
import org.jfree.data.xy.XYSeries;
import org.jfree.data.xy.XYSeriesCollection;
import org.jfree.ui.ApplicationFrame;

// TODO: Auto-generated Javadoc
/**
 * Klasa reprezentujaca wykres kosztow i wydatkow rysowany w menu administratora.
 */
@SuppressWarnings("serial")
public class Chart extends ApplicationFrame {
	
	private String title;
	private ArrayList<Number> data_x;
	private ArrayList<Number> data_y;
	
	/**
	 * Tworzy nowy wykres z konkretnymi parametrami.
	 *
	 * @param title tytul wykresu
	 * @param data_x wartosci na osi X
	 * @param data_y wartosci na osi Y
	 * @throws Exception the exception
	 */
	public Chart(String title, ArrayList<Number> data_x, ArrayList<Number> data_y) throws Exception {
		super(title);
		this.title = title;
		this.data_x = data_x;
		this.data_y = data_y;
		
		if(data_x.size() != data_y.size()) throw new Exception("X and Y arraylists's sizes are not equal!");
		
		final ChartPanel chartPanel = new ChartPanel(generateChart());
		chartPanel.setPreferredSize(new Dimension(1280,480));
		setContentPane(chartPanel);
		
		this.pack();
		//RefineryUtilities.centerFrameOnScreen(this);
		this.setVisible(true);
	}
	
	/**
	 * Generuje wykres jako element interfejsu graficznego.
	 *
	 * @return Wykres w postaci JFreeChart.
	 */
	private JFreeChart generateChart()
	{
		final XYSeries series = new XYSeries("Test");
		for(int i = 0; i< data_x.size(); i++)
		{
			series.add(data_x.get(i), data_y.get(i));
			//System.out.println(String.valueOf(data_x.get(i)) + " | " + String.valueOf(data_y.get(i)));
		}
		final XYSeriesCollection data = new XYSeriesCollection(series);
		final JFreeChart chart = ChartFactory.createTimeSeriesChart(
				title, 
				"CZAS", 
				"PRZYCH. / WYD.", 
				data, 
				false, 
				false, 
				false
		);
		chart.setBackgroundPaint(new Paint() {

			@Override
			public int getTransparency() {
				return Transparency.OPAQUE;
			}

			@Override
			public PaintContext createContext(ColorModel cm,
					Rectangle deviceBounds, Rectangle2D userBounds,
					AffineTransform xform, RenderingHints hints) {
				return new myPaintContext(cm, xform);
			}
			
		});
		
		return chart;
	}
	
	
	class myPaintContext implements PaintContext {

		public myPaintContext(ColorModel cm_, AffineTransform xform_) { }
		
		/* (non-Javadoc)
		 * @see java.awt.PaintContext#dispose()
		 */
		@Override
		public void dispose() 
		{
			// TODO Auto-generated method stub
			
		}

		/* (non-Javadoc)
		 * @see java.awt.PaintContext#getColorModel()
		 */
		@Override
		public ColorModel getColorModel() 
		{
			return ColorModel.getRGBdefault();
		}

		/* (non-Javadoc)
		 * @see java.awt.PaintContext#getRaster(int, int, int, int)
		 */
		@Override
		public Raster getRaster(int x, int y, int w, int h) 
		{
			WritableRaster raster = getColorModel().createCompatibleWritableRaster(w, h);
			float[] colors = {255, 255, 255, 255};
			for(int i = 0; i < h; i++)
			{
				for(int j = 0; j < w; j++)
				{
					raster.setPixel(j, i, colors);
				}
			}
			return raster;
	}
	}
}
