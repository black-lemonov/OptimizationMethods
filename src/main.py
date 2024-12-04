from app_builder import Builder, Director


if __name__ == '__main__':
    builder = Builder()
    director = Director()
    director.maximum_app(builder).run()
    