from app import *
from funcs import *
import app_builder as ab


if __name__ == '__main__':
    themed_tk_builder = ab.ThemedTkAppBuilder()
    builder_director = ab.AppBuilderDirector()
    builder_director.maximum_app(themed_tk_builder).run()
    