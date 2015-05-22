package  {
	import away3d.textures.VideoTexture;
	import away3d.materials.utils.IVideoPlayer;
	import away3d.textures.BitmapTexture;
	import flash.display.Bitmap;
	import flash.display.BitmapData;
	import flash.geom.Matrix;
	import flash.geom.Rectangle;
	import flash.geom.Point;
	import flash.display.BlendMode;
	
	public class CustomVideoTexture extends VideoTexture 
	{	
		private var playTex:BitmapTexture;
		private var playBM:BitmapData;
		private var thumbBM:BitmapData;
		private var pausedBM:BitmapData;
		private var lastBM:BitmapData;
		
		public function CustomVideoTexture(source:String, materialWidth:uint, materialHeight:uint, loop:Boolean,
										   autoPlay:Boolean, thumb:BitmapData, player:IVideoPlayer)
		{
			super(source, materialWidth, materialHeight, loop, autoPlay, player);
			
			this.thumbBM = thumb;
			
			GenerateStartScreen();
		}
		
		public function play() : void
		{
			this.bitmapData = this.lastBM;
			this.player.play();
		}
		
		public function pause() : void
		{
			GeneratePauseScreen();
			this.player.pause();
		}
		
		public function stop() : void
		{
			this.lastBM = this.bitmapData;
			this.bitmapData = this.thumbBM;
			this.player.stop();
		}
		
		private function GenerateStartScreen() : void
		{
			this.playTex = System.getInstance().Textures["textures/PlayVideoTexture.png"];
			var sbm:BitmapData = playTex.bitmapData;
			var nbm:BitmapData = new BitmapData(512, 512, true, 0);
			var mt:Matrix = new Matrix();
			mt.scale(0.5625 / 2, 1 / 2);
			mt.translate(nbm.width / 2 - sbm.width * 0.5625 / 4 , nbm.height / 2 - sbm.height / 4);
			nbm.draw(sbm, mt);
			
			
			this.playBM = nbm;
			this.lastBM = this.bitmapData;
			
			var newT:BitmapData = new BitmapData(512, 512, true, 0);
			var mtt:Matrix = new Matrix();
			mtt.scale(5/4, 3.75/4);
			newT.draw(thumbBM, mtt);
			newT.merge(this.playBM, new Rectangle(0, 0, 512, 512), new Point(0, 0), 128, 128, 128, 0);
			this.thumbBM = newT;
			this.bitmapData = this.thumbBM;
		}
		
		private function GeneratePauseScreen() : void
		{
			this.lastBM = this.bitmapData;
			
			var newT:BitmapData = new BitmapData(512, 512, true, 0);
			var mtt:Matrix = new Matrix();
			//mtt.scale(5, 3.75);
			newT.draw(this.lastBM, mtt);
			newT.merge(this.playBM, new Rectangle(0, 0, 512, 512), new Point(0, 0), 128, 128, 128, 0);
			this.pausedBM = newT;
			this.bitmapData = this.pausedBM;
		}
	}
	
}
